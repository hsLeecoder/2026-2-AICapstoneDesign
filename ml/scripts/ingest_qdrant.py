"""학습된 임베딩 모델로 청크를 벡터로 바꿔 Qdrant에 저장한다 (M4 벡터 적재).

사용 예 (EC2 터미널, 프로젝트 루트에서, (pytorch) 환경 켠 상태):
    python ml/scripts/ingest_qdrant.py --chunks ml/data/chunks/chunks.jsonl \
        --model ml/output/bge-m3-finetuned/final --recreate \
        --query "배우자 출산휴가는 며칠인가요?"
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
    p = argparse.ArgumentParser(description="청크를 임베딩해서 Qdrant에 적재")
    p.add_argument("--chunks", required=True, help="청크 .jsonl (prepare_chunks.py 결과)")
    p.add_argument("--model", default="ml/output/bge-m3-finetuned/final", help="임베딩 모델 이름 또는 폴더 경로")
    p.add_argument("--collection", default="welfare_chunks_bgem3", help="Qdrant 컬렉션 이름")
    p.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant 주소")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--recreate", action="store_true", help="같은 이름의 컬렉션이 있으면 지우고 새로 만듦")
    p.add_argument("--query", default=None, help="적재 후 검색이 되는지 확인할 질문 (선택)")
    a = p.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    chunks = read_jsonl(a.chunks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[시작] 모델: {a.model} / 장치: {device} / 청크 {len(chunks)}개 -> 컬렉션 '{a.collection}'")

    model = SentenceTransformer(a.model, device=device)
    model.max_seq_length = a.max_seq_length
    dim = model.get_sentence_embedding_dimension()
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=a.batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"[임베딩 완료] 벡터 {len(vectors)}개, 차원 {dim}")

    client = QdrantClient(url=a.qdrant_url)
    if client.collection_exists(a.collection):
        if a.recreate:
            client.delete_collection(a.collection)
        else:
            raise SystemExit(f"[중단] 컬렉션 '{a.collection}'이 이미 있습니다. 덮어쓰려면 --recreate 를 붙이세요.")
    client.create_collection(a.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    points = [
        PointStruct(
            id=i,  # Qdrant의 id는 정수 또는 UUID만 가능해서 순번을 쓰고, 원래 id는 payload에 둔다
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": c["chunk_id"],
                "doc": c["doc"],
                "page": c["page"],
                "heading": c.get("heading", ""),
                "text": c["text"],
            },
        )
        for i, c in enumerate(chunks)
    ]
    for s in range(0, len(points), 100):
        client.upsert(collection_name=a.collection, points=points[s:s + 100])
    print(f"[적재 완료] 컬렉션 '{a.collection}' 저장된 개수: {client.count(a.collection).count}")

    if a.query:
        qv = model.encode([a.query], normalize_embeddings=True)[0].tolist()
        res = client.query_points(a.collection, query=qv, limit=3, with_payload=True).points
        print(f"\n[검색 확인] 질문: {a.query}")
        for r in res:
            pl = r.payload
            print(f"  점수 {r.score:.3f} | {pl['doc']} p.{pl['page']} | {pl['text'][:60]}...")


if __name__ == "__main__":
    main()
