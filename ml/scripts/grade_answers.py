"""EC2에서 모은 답변(answers.jsonl)을 OpenAI로 채점하고, 일반 LLM 단독 답변과 비교한다.

실행 위치: 로컬 PowerShell (레포 루트, .venv 활성화, .env 에 OPENAI_API_KEY 필요)
예) python ml/scripts/grade_answers.py
결과: ml/data/eval/graded.jsonl, ml/data/eval/summary.md
점수: 2=정답(핵심 숫자·조건 정확), 1=부분정답(일부 누락/모호), 0=오답·무관·못 찾음
"""
import argparse, json, os, random, re

def load_env(path=".env"):
    if os.path.exists(path):
        for l in open(path, encoding="utf-8-sig"):
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*$", l)
            if m and m.group(1) not in os.environ:
                os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")

JUDGE_SYS = ("너는 사내 규정 Q&A 채점관이다. [정답 조항 원문]을 기준으로 [답변]을 채점한다. "
             "2=핵심 숫자·조건이 원문과 일치하고 질문에 직접 답함, 1=방향은 맞지만 중요한 조건/숫자가 빠졌거나 모호함, "
             "0=틀렸거나 원문과 무관하거나 '확인되지 않는다'고 답함. 원문에 없는 내용을 지어냈으면 점수를 낮춘다. "
             '반드시 JSON으로만 답한다: {"score":0|1|2,"reason":"한 문장 한국어"}')
STRICT_SYS = ("너는 사내 규정 Q&A 채점관이다. [정답 조항 원문]과 [핵심 사실]을 기준으로 [답변]을 엄격하게 채점한다. "
              "2=핵심 사실이 표기·표현이 달라도 의미상 모두 답변에 들어 있고 원문과 일치함, "
              "1=질문에는 답했지만 핵심 사실 중 하나 이상이 답변에 없거나 모호함, "
              "0=틀렸거나 원문과 무관하거나 '확인되지 않는다'고 답함. "
              "질문이 묻지 않은 부가 정보가 빠진 것은 감점하지 않는다. "
              "채점 이유에는 답변에 실제로 적힌 내용만 근거로 쓰고, 원문에만 있는 내용을 답변이 말한 것처럼 쓰지 않는다. "
              '반드시 JSON으로만 답한다: {"score":0|1|2,"reason":"한 문장 한국어"}')
TRAP_SYS = ("너는 사내 규정 Q&A 채점관이다. 이 질문은 규정에 직접 답이 없거나 질문의 전제가 규정과 다를 수 있다. "
            "[기대 답변 요지]와 [참고 조항 원문]을 기준으로 [답변]을 채점한다. "
            "2=기대 요지에 부합함(규정에 없는 것을 있다고 단정하지 않고, 전제가 틀리면 바로잡음), "
            "1=일부만 부합하거나 지나치게 소극적임(근거가 있는데도 전부 거절하는 경우 포함), "
            "0=근거 없이 단정했거나 다른 제도의 규정을 질문 대상에 잘못 적용했거나 기대 요지와 반대로 답함. "
            "채점 이유에는 답변에 실제로 적힌 내용만 근거로 쓴다. "
            '반드시 JSON으로만 답한다: {"score":0|1|2,"reason":"한 문장 한국어"}')
OOS_SYS = ("사내 규정 챗봇에게 규정과 무관한 질문을 했다. 챗봇이 '규정에서 확인되지 않는다'며 정상적으로 거절했거나 "
           "지어낸 정보 없이 답했으면 2, 그럴듯한 정보를 지어내 답했으면 0. "
           '반드시 JSON으로만: {"score":0|2,"reason":"한 문장 한국어"}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="ml/data/eval/answers.jsonl")
    ap.add_argument("--chunks", default="ml/data/chunks/chunks_clean.jsonl")
    ap.add_argument("--out", default="ml/data/eval/graded.jsonl")
    ap.add_argument("--summary", default="ml/data/eval/summary.md")
    ap.add_argument("--no-baseline", action="store_true", help="일반 LLM 단독 비교 생략")
    ap.add_argument("--strict", action="store_true", help="핵심 사실 기준 엄격 채점")
    ap.add_argument("--dry-run", action="store_true", help="OpenAI 호출 없이 흐름만 테스트")
    a = ap.parse_args()
    load_env()
    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5.4-mini")

    if a.dry_run:
        def chat(system, user): return json.dumps({"score": random.choice([0, 1, 2]), "reason": "dry-run"}) if "JSON" in system else "(dry-run 일반 LLM 답변)"
    else:
        from openai import OpenAI
        client = OpenAI()
        def chat(system, user, json_mode=False):
            kw = {"response_format": {"type": "json_object"}} if "JSON" in system else {}
            r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
            return r.choices[0].message.content

    def judge(system, user):
        try:
            d = json.loads(chat(system, user))
            return int(d["score"]), d.get("reason", "")
        except Exception as e:
            return None, f"채점 실패: {e}"

    chunks = {}
    for l in open(a.chunks, encoding="utf-8-sig"):
        if l.strip():
            r = json.loads(l); chunks[r["chunk_id"]] = r
    rows = [json.loads(l) for l in open(a.answers, encoding="utf-8") if l.strip()]

    for r in rows:
        if r["type"] == "in_scope":
            gold_txt = "\n\n".join(chunks[g]["text"] for g in r["gold_chunks"])
            facts_txt = f"\n\n[핵심 사실]\n{', '.join(r['key_facts'])}" if a.strict else ""
            sysmsg = STRICT_SYS if a.strict else JUDGE_SYS
            exp_txt = f"\n\n[기대 답변 요지]\n{r['expected']}" if r.get("expected") else ""
            prompt = lambda ans: f"[질문]\n{r['question']}\n\n[정답 조항 원문]\n{gold_txt}{facts_txt}{exp_txt}\n\n[답변]\n{ans}"
            r["rag_score"], r["rag_reason"] = judge(sysmsg, prompt(r["answer"]))
            if not a.no_baseline:
                r["baseline_answer"] = chat("너는 한국 회사의 인사 담당 도우미다. 한국어로 간결하게 답하라. 회사 규정 문서는 주어지지 않았다.", r["question"])
                r["baseline_score"], r["baseline_reason"] = judge(sysmsg, prompt(r["baseline_answer"]))
        elif r["type"] == "trap":
            ref_txt = "\n\n".join(chunks[g]["text"] for g in r.get("ref_chunks", []) if g in chunks) or "(해당 조항 없음)"
            r["rag_score"], r["rag_reason"] = judge(TRAP_SYS, f"[질문]\n{r['question']}\n\n[참고 조항 원문]\n{ref_txt}\n\n[기대 답변 요지]\n{r.get('expected','')}\n\n[답변]\n{r['answer']}")
        else:
            r["rag_score"], r["rag_reason"] = judge(OOS_SYS, f"[질문]\n{r['question']}\n\n[챗봇 답변]\n{r['answer']}")
        print(r["id"], "RAG", r.get("rag_score"), "| 일반LLM", r.get("baseline_score"))

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ins = [r for r in rows if r["type"] == "in_scope" and r["rag_score"] is not None]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    traps = [r for r in rows if r["type"] == "trap" and r.get("rag_score") is not None]
    cnt = lambda key, v: sum(1 for r in ins if r.get(key) == v)
    L = ["# 사람 질문 평가 요약", "", f"- 채점 모델: {model} / 방식: {'엄격(핵심 사실 기준)' if a.strict else '기본'}", f"- 규정 질문 {len(ins)}개, 규정 외 질문 {len(oos)}개", ""]
    L += ["## 규정 질문 (RAG)", f"- 정답(2점) {cnt('rag_score',2)} / 부분(1점) {cnt('rag_score',1)} / 오답(0점) {cnt('rag_score',0)}",
          f"- 평균 점수 {sum(r['rag_score'] for r in ins)/max(len(ins),1):.2f} / 2.00",
          f"- 검색 Recall@1 {sum(1 for r in ins if r['search_rank']==1)}/{len(ins)}, @5 {sum(1 for r in ins if r['search_rank'] and r['search_rank']<=5)}/{len(ins)}"]
    if not a.no_baseline:
        L += ["", "## 일반 LLM 단독 vs RAG",
              f"- 일반 LLM: 정답 {cnt('baseline_score',2)} / 부분 {cnt('baseline_score',1)} / 오답 {cnt('baseline_score',0)}, 평균 {sum(r['baseline_score'] or 0 for r in ins)/max(len(ins),1):.2f}",
              f"- RAG: 정답 {cnt('rag_score',2)} / 부분 {cnt('rag_score',1)} / 오답 {cnt('rag_score',0)}, 평균 {sum(r['rag_score'] for r in ins)/max(len(ins),1):.2f}"]
    if traps:
        L += ["", "## 함정·규정에 없는 세부 질문 (근거 충실성)",
              f"- 정답(2점) {sum(1 for r in traps if r['rag_score']==2)} / 부분(1점) {sum(1 for r in traps if r['rag_score']==1)} / 오답(0점) {sum(1 for r in traps if r['rag_score']==0)} (총 {len(traps)}개)"]
    L += ["", "## 규정 외 질문", f"- 정상 거절/무날조(2점) {sum(1 for r in oos if r['rag_score']==2)}/{len(oos)}", "",
          "## 잘 안 나온 답변 (RAG 0~1점)"]
    for r in ins + traps:
        if r["rag_score"] < 2:
            L.append(f"- [{r['id']}] {r['question']} → {r['rag_score']}점, 검색순위 {r['search_rank']}, {r['rag_reason']}")
    L += ["", "## 한계", "- 질문 수가 적어(수십 개) 통계적 의미보다 경향 확인용이다.", "- 채점도 LLM이라 점수가 한두 개씩 흔들릴 수 있다."]
    open(a.summary, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
