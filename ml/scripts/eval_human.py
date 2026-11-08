"""사람이 쓴 평가 질문으로 검색 서버(/search, /answer)를 평가하고 답변을 파일로 저장한다.

실행 위치: EC2 브라우저 터미널 (검색 서버가 켜져 있어야 함, USE_LLM=1)
예) python ml/scripts/eval_human.py
결과: ml/data/eval/answers.jsonl  → 로컬로 내려받아 grade_answers.py 로 채점
"""
import argparse, json, re, time
import requests

QKEYS = ["question", "query", "q", "text"]
ANS_KEYS = ["answer", "response", "text", "output"]
SRC_KEYS = ["sources", "results", "hits", "contexts", "evidence", "citations", "references"]
def norm(s): return re.sub(r"\s+", "", s or "")

def build_body(props, question, extras):
    """서버의 /openapi.json 을 보고 요청 필드 이름을 자동으로 맞춘다."""
    key = next((k for k in QKEYS if k in props), None)
    if key is None:
        key = next((k for k, v in props.items() if v.get("type") == "string"), "question")
    body = {key: question}
    for k, v in extras.items():
        if k in props:
            body[k] = v
    return body

def get_props(base, path):
    try:
        spec = requests.get(f"{base}/openapi.json", timeout=10).json()
        schema = spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
        if "$ref" in schema:
            schema = spec["components"]["schemas"][schema["$ref"].split("/")[-1]]
        return schema.get("properties", {})
    except Exception as e:
        print(f"[경고] {path} 스키마 자동 확인 실패({e}) → 기본 필드 'question' 사용")
        return {"question": {"type": "string"}}

def flat(item):
    d = dict(item) if isinstance(item, dict) else {}
    for k in ("payload", "metadata", "chunk"):
        if isinstance(d.get(k), dict):
            d = {**d.pop(k), **d}
    return d

def find_sources(resp):
    if isinstance(resp, list):
        return [flat(x) for x in resp]
    for k in SRC_KEYS:
        if isinstance(resp.get(k), list):
            return [flat(x) for x in resp[k]]
    return []

def find_answer(resp):
    if not isinstance(resp, dict): return ""
    for k in ANS_KEYS:
        if isinstance(resp.get(k), str): return resp[k]
    return ""

def is_gold(src, gold_ids, gold_meta):
    if src.get("chunk_id") in gold_ids: return True
    return any(src.get("heading") == h and src.get("page") == p for h, p in gold_meta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", default="ml/data/eval/human_questions.jsonl")
    ap.add_argument("--chunks", default="ml/data/chunks/chunks_clean.jsonl")
    ap.add_argument("--out", default="ml/data/eval/answers.jsonl")
    a = ap.parse_args()

    chunks = {}
    for l in open(a.chunks, encoding="utf-8-sig"):
        if l.strip():
            r = json.loads(l); chunks[r["chunk_id"]] = r
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8-sig") if l.strip()]

    sp, ap_ = get_props(a.base, "/search"), get_props(a.base, "/answer")
    print("/search 필드:", list(sp), "| /answer 필드:", list(ap_))
    s_extra = {"rerank": True, "candidates": 20, "top_k": 5, "k": 5, "limit": 5}
    a_extra = {"rerank": True, "candidates": 20}

    out, t0 = [], time.time()
    for q in qs:
        gold = q["gold_chunks"]
        gmeta = [(chunks[g]["heading"], chunks[g]["page"]) for g in gold]
        # 1) 검색 순위
        sr = requests.post(f"{a.base}/search", json=build_body(sp, q["question"], s_extra), timeout=120).json()
        s_src = find_sources(sr)
        rank = next((i + 1 for i, s in enumerate(s_src) if is_gold(s, gold, gmeta)), None)
        # 2) 답변 생성
        t = time.time()
        ar = requests.post(f"{a.base}/answer", json=build_body(ap_, q["question"], a_extra), timeout=600).json()
        sec = round(time.time() - t, 1)
        ans, a_src = find_answer(ar), find_sources(ar)
        facts = q["key_facts"]
        fact_hit = [f for f in facts if norm(f) in norm(ans)]
        out.append({
            "id": q["id"], "question": q["question"], "type": q["type"],
            "gold_chunks": gold, "key_facts": facts,
            "search_rank": rank,
            "answer": ans,
            "answer_sources": [{k: s.get(k) for k in ("chunk_id", "doc", "heading", "page", "rerank_score", "score") if k in s} for s in a_src],
            "source_hit": any(is_gold(s, gold, gmeta) for s in a_src) if gold else None,
            "refused": "확인되지 않" in ans,
            "key_fact_hits": fact_hit,
            "key_fact_ratio": round(len(fact_hit) / len(facts), 2) if facts else None,
            "answer_sec": sec,
            "raw_answer_keys": list(ar.keys()) if isinstance(ar, dict) else "list",
        })
        print(f"{q['id']} rank={rank} refused={out[-1]['refused']} facts={out[-1]['key_fact_ratio']} {sec}s")

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ins = [r for r in out if r["type"] == "in_scope"]; oos = [r for r in out if r["type"] == "out_of_scope"]
    n = len(ins)
    print("\n===== 요약 =====")
    for k in (1, 3, 5):
        print(f"검색 Recall@{k}: {sum(1 for r in ins if r['search_rank'] and r['search_rank'] <= k)}/{n}")
    print(f"답변 출처에 정답 조항 포함: {sum(1 for r in ins if r['source_hit'])}/{n}")
    print(f"규정 질문인데 '확인되지 않습니다'로 거절: {sum(1 for r in ins if r['refused'])}/{n}")
    fr = [r['key_fact_ratio'] for r in ins if not r['refused']]
    print(f"핵심 사실 포함률(거절 제외 평균): {sum(fr)/len(fr):.2f}" if fr else "핵심 사실 포함률: -")
    print(f"규정 외 질문 정상 거절: {sum(1 for r in oos if r['refused'])}/{len(oos)}")
    print(f"총 소요 {time.time()-t0:.0f}초 → {a.out}")

if __name__ == "__main__":
    main()
