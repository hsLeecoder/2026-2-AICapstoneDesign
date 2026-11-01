"""M7 리랭커 효과를 잰다: "검색만" vs "검색 + 리랭커" (Recall@k, MRR).

1) 임베딩 모델로 청크 전체를 검색해 상위 N개(기본 20개)를 후보로 뽑고,
2) 리랭커가 질문과 후보를 하나씩 대조해 다시 정렬한 뒤,
3) 정답 청크의 순위를 두 방식으로 비교한다.
리랭커를 따로 학습하지 않고 공개 모델(BAAI/bge-reranker-v2-m3)을 그대로 쓴다.

사용 예 (EC2 터미널, 프로젝트 루트에서, (pytorch) 환경 켠 상태):
    python ml/scripts/eval_rerank.py --chunks ml/data/chunks/chunks_dedup.jsonl \
        --valid ml/data/pairs/valid_dedup.jsonl --output ml/output/eval_rerank_v1.json
"""
import argparse
import json

import numpy as np


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics(ranks):
    """ranks: 정답 청크의 순위 배열 (1등이면 1, 후보에 없으면 inf)"""
    r = np.asarray(ranks, dtype=float)
    return {
        "recall@1": float((r <= 1).mean()),
        "recall@5": float((r <= 5).mean()),
        "recall@10": float((r <= 10).mean()),
        "mrr": float(np.where(np.isfinite(r), 1.0 / r, 0.0).mean()),
    }


def main():
    p = argparse.ArgumentParser(description="리랭커 적용 전·후 검색 성능 비교")
    p.add_argument("--chunks", required=True, help="청크 .jsonl")
    p.add_argument("--valid", required=True, help="검증용 문제집 .jsonl")
    p.add_argument("--embed-model", default="ml/output/bge-m3-finetuned/final", help="임베딩 모델 이름 또는 폴더")
    p.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3", help="리랭커 모델 이름 또는 폴더")
    p.add_argument("--top-n", type=int, default=20, help="리랭커에 넘길 후보 개수")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default=None, help="결과를 저장할 .json 경로 (선택)")
    a = p.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    chunks = read_jsonl(a.chunks)
    valid = read_jsonl(a.valid)
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    id2idx = {cid: i for i, cid in enumerate(ids)}
    valid = [v for v in valid if v["chunk_id"] in id2idx]
    print(f"[시작] 청크 {len(ids)}개 / 검증 질문 {len(valid)}개 / 후보 {a.top_n}개 / 장치 {device}")

    # 1) 임베딩 검색 (리랭커 적용 전)
    emb = SentenceTransformer(a.embed_model, device=device)
    emb.max_seq_length = a.max_seq_length
    enc = dict(batch_size=a.batch_size, normalize_embeddings=True, show_progress_bar=False)
    doc_emb = np.asarray(emb.encode(texts, **enc))
    q_emb = np.asarray(emb.encode([v["query"] for v in valid], **enc))
    sims = q_emb @ doc_emb.T
    gold = np.array([id2idx[v["chunk_id"]] for v in valid])
    gold_scores = sims[np.arange(len(valid)), gold][:, None]
    base_ranks = (sims > gold_scores).sum(axis=1) + 1

    # 2) 리랭커로 후보 재정렬
    n = min(a.top_n, len(ids))
    cand = np.argsort(-sims, axis=1)[:, :n]
    reranker = CrossEncoder(a.reranker, device=device, max_length=a.max_seq_length)
    pairs = [(v["query"], texts[j]) for v, row in zip(valid, cand) for j in row]
    scores = np.asarray(reranker.predict(pairs, batch_size=a.batch_size, show_progress_bar=False)).reshape(len(valid), n)

    rerank_ranks = []
    for qi in range(len(valid)):
        order = cand[qi][np.argsort(-scores[qi])]       # 리랭커 점수 높은 순서의 청크 번호
        pos = np.where(order == gold[qi])[0]
        rerank_ranks.append(pos[0] + 1 if len(pos) else np.inf)

    base = metrics(base_ranks)
    rer = metrics(rerank_ranks)
    in_cand = float((base_ranks <= n).mean())

    print(f"\n=== 검색 성능 비교 (검증 질문 {len(valid)}개) ===")
    print(f"  {'지표':<10}{'검색만':>10}{'검색+리랭커':>14}{'변화':>10}")
    for k in ("recall@1", "recall@5", "recall@10", "mrr"):
        print(f"  {k:<10}{base[k]:>10.3f}{rer[k]:>14.3f}{rer[k] - base[k]:>+10.3f}")
    print(f"\n  후보 {n}개 안에 정답이 들어 있는 비율: {in_cand:.3f}  (리랭커가 올릴 수 있는 최대치)")

    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump({"embed_model": a.embed_model, "reranker": a.reranker, "top_n": n,
                       "n_queries": len(valid), "n_chunks": len(ids),
                       "retrieval_only": base, "with_reranker": rer, "gold_in_candidates": in_cand},
                      f, ensure_ascii=False, indent=2)
        print(f"[저장] {a.output}")


if __name__ == "__main__":
    main()
