"""채점 결과(graded jsonl)에서 의심스러운 답변을 골라 본다.
- 핵심 사실 자동 체크에서 누락이 있는데 OpenAI 채점은 2점인 경우 (채점이 후한지 확인)
- RAG 점수가 2점 미만인 경우
실행 위치: 로컬 PowerShell (레포 루트)
예) python ml/scripts/inspect_graded.py --graded ml/data/eval/graded_min0.005.jsonl
"""
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", default="ml/data/eval/graded.jsonl")
    ap.add_argument("--chunks", default="ml/data/chunks/chunks_clean.jsonl")
    ap.add_argument("--max-chars", type=int, default=350)
    a = ap.parse_args()
    chunks = {}
    for l in open(a.chunks, encoding="utf-8-sig"):
        if l.strip():
            r = json.loads(l); chunks[r["chunk_id"]] = r
    rows = [json.loads(l) for l in open(a.graded, encoding="utf-8") if l.strip()]
    ins = [r for r in rows if r["type"] == "in_scope"]
    miss = [r for r in ins if r.get("key_fact_ratio") is not None and r["key_fact_ratio"] < 1]
    low = [r for r in ins if (r.get("rag_score") or 0) < 2]
    lenient = [r for r in miss if r.get("rag_score") == 2]
    print(f"규정 질문 {len(ins)}개 | 핵심 사실 누락 {len(miss)}개 | 그중 채점 2점 {len(lenient)}개 | RAG 2점 미만 {len(low)}개\n")
    seen = set()
    for r in miss + [x for x in low if x not in miss]:
        if r["id"] in seen: continue
        seen.add(r["id"])
        missing = [f for f in r["key_facts"] if f not in r["key_fact_hits"]]
        print("=" * 70)
        print(f"[{r['id']}] {r['question']}")
        print(f"  RAG 점수 {r.get('rag_score')} / 누락된 핵심 사실 {missing} / 검색순위 {r['search_rank']}")
        print(f"  채점 이유: {r.get('rag_reason')}")
        print(f"  답변: {r['answer'][:a.max_chars].strip()}")
        g = chunks[r["gold_chunks"][0]]
        print(f"  정답 조항: {g['heading']} (p.{g['page']})")
    if not seen:
        print("의심스러운 답변 없음")

if __name__ == "__main__":
    main()
