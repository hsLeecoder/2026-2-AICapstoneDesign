"""임베딩 모델의 검색 성능을 잰다 (Recall@k, MRR).

검증용 질문을 임베딩해서 청크 전체(301개) 중 정답 청크가 몇 등으로 검색되는지 본다.
학습 전 모델(BAAI/bge-m3)과 학습 후 모델을 같은 검증 질문으로 비교하는 용도.

사용 예 (EC2 터미널, 프로젝트 루트에서, 가상환경 켠 상태):
    python ml/scripts/eval_retrieval.py --chunks ml/data/chunks/chunks.jsonl \
        --valid ml/data/pairs/valid.jsonl --model BAAI/bge-m3 --label base
"""
import argparse
import json


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser(description="임베딩 검색 성능 평가")
    p.add_argument("--chunks", required=True, help="청크 .jsonl (prepare_chunks.py 결과)")
    p.add_argument("--valid", required=True, help="검증용 문제집 .jsonl")
    p.add_argument("--model", default="BAAI/bge-m3", help="모델 이름 또는 학습된 모델 폴더 경로")
    p.add_argument("--max-seq-length", type=int, default=512, help="한 번에 읽는 최대 토큰 길이")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--label", default="", help="결과에 붙일 이름 (예: base, finetuned)")
    p.add_argument("--output", default=None, help="결과를 저장할 .json 경로 (선택)")
    a = p.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    chunks = read_jsonl(a.chunks)
    valid = read_jsonl(a.valid)
    ids = [c["chunk_id"] for c in chunks]
    id2idx = {cid: i for i, cid in enumerate(ids)}

    skipped = [v for v in valid if v["chunk_id"] not in id2idx]
    valid = [v for v in valid if v["chunk_id"] in id2idx]
    if skipped:
        print(f"[경고] 청크 파일에 없는 chunk_id 때문에 질문 {len(skipped)}개를 건너뜀")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[시작] 모델: {a.model} / 장치: {device} / 청크 {len(ids)}개 / 검증 질문 {len(valid)}개")
    model = SentenceTransformer(a.model, device=device)
    model.max_seq_length = a.max_seq_length

    enc = dict(batch_size=a.batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    doc_emb = model.encode([c["text"] for c in chunks], **enc)
    q_emb = model.encode([v["query"] for v in valid], **enc)

    sims = q_emb @ doc_emb.T                                   # (질문 수, 청크 수) 코사인 유사도
    gold = torch.tensor([id2idx[v["chunk_id"]] for v in valid], device=sims.device)
    gold_scores = sims.gather(1, gold.unsqueeze(1))
    ranks = ((sims > gold_scores).sum(dim=1) + 1).float()      # 정답 청크의 순위 (1등이면 1)

    result = {
        "label": a.label,
        "model": a.model,
        "n_queries": len(valid),
        "n_chunks": len(ids),
        "recall@1": (ranks <= 1).float().mean().item(),
        "recall@5": (ranks <= 5).float().mean().item(),
        "recall@10": (ranks <= 10).float().mean().item(),
        "mrr": (1.0 / ranks).mean().item(),
    }
    print("\n=== 검색 성능 ===")
    print(f"  Recall@1  : {result['recall@1']:.3f}")
    print(f"  Recall@5  : {result['recall@5']:.3f}")
    print(f"  Recall@10 : {result['recall@10']:.3f}")
    print(f"  MRR       : {result['mrr']:.3f}")

    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[저장] {a.output}")


if __name__ == "__main__":
    main()
