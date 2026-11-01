"""질문을 벡터로 바꾸고(M5) Qdrant에서 비슷한 청크를 찾은 뒤(M6) 리랭커로 다시 정렬(M7)하는 FastAPI 서버.

실행 (EC2 터미널, 프로젝트 루트에서, (pytorch) 환경 켠 상태):
    nohup python -m uvicorn --app-dir ml/scripts search_server:app --host 127.0.0.1 --port 8000 > ml/output/server.log 2>&1 &

설정은 환경변수로 바꿀 수 있다 (안 정하면 아래 기본값 사용):
    MODEL_PATH     (기본 ml/output/bge-m3-finetuned/final)   임베딩 모델
    RERANKER_PATH  (기본 BAAI/bge-reranker-v2-m3)             리랭커 모델
    USE_RERANKER   (기본 1, 끄려면 0)                          리랭커 사용 여부
    QDRANT_URL     (기본 http://localhost:6333)
    COLLECTION     (기본 welfare_chunks_bgem3)
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

state = {}  # 서버가 켜질 때 model, reranker, client, collection 을 채운다


@asynccontextmanager
async def lifespan(app):
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from qdrant_client import QdrantClient

    model_path = os.environ.get("MODEL_PATH", "ml/output/bge-m3-finetuned/final")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    model.max_seq_length = 512
    state["model"] = model
    state["model_path"] = model_path

    state["reranker"] = None
    state["reranker_path"] = None
    if os.environ.get("USE_RERANKER", "1") == "1":
        reranker_path = os.environ.get("RERANKER_PATH", "BAAI/bge-reranker-v2-m3")
        state["reranker"] = CrossEncoder(reranker_path, device=device, max_length=512)
        state["reranker_path"] = reranker_path

    state["client"] = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    state["collection"] = os.environ.get("COLLECTION", "welfare_chunks_bgem3")
    print(f"[서버 준비 완료] 임베딩: {model_path} / 리랭커: {state['reranker_path']} / 장치: {device} / 컬렉션: {state['collection']}")
    yield
    state.clear()


app = FastAPI(title="규정 검색 서버 (M5 질의 임베딩 + M6 벡터 검색 + M7 리랭킹)", lifespan=lifespan)


class EmbedRequest(BaseModel):
    question: str = Field(min_length=1, description="직원의 질문")


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, description="직원의 질문")
    top_k: int = Field(5, ge=1, le=20, description="돌려줄 청크 개수")
    rerank: bool = Field(True, description="리랭커로 재정렬할지 (서버에 리랭커가 없으면 무시)")
    candidates: int = Field(20, ge=1, le=50, description="리랭커에 넘길 후보 개수")


def embed(question: str):
    return state["model"].encode([question], normalize_embeddings=True)[0].tolist()


@app.get("/health")
def health():
    count = state["client"].count(state["collection"]).count
    return {"status": "ok", "model": state["model_path"], "reranker": state["reranker_path"],
            "collection": state["collection"], "chunks": count}


@app.post("/embed")
def embed_endpoint(req: EmbedRequest):
    """M5 질의 임베딩: 질문(문자열) -> 질문 벡터"""
    vec = embed(req.question)
    return {"dim": len(vec), "vector": vec}


@app.post("/search")
def search(req: SearchRequest):
    """M5 + M6 (+ M7): 질문 -> 상위 k개 청크 (본문, 출처, 점수). 리랭커를 쓰면 rerank_score도 함께 반환"""
    use_rerank = req.rerank and state.get("reranker") is not None
    limit = max(req.top_k, req.candidates) if use_rerank else req.top_k
    points = state["client"].query_points(
        state["collection"], query=embed(req.question), limit=limit, with_payload=True
    ).points

    rerank_scores = None
    if use_rerank:
        pairs = [(req.question, p.payload["text"]) for p in points]
        rerank_scores = [float(s) for s in state["reranker"].predict(pairs, show_progress_bar=False)]
        order = sorted(range(len(points)), key=lambda i: -rerank_scores[i])[: req.top_k]
    else:
        order = list(range(len(points)))[: req.top_k]

    results = []
    for i in order:
        p = points[i]
        item = {
            "score": round(p.score, 4),
            "chunk_id": p.payload["chunk_id"],
            "doc": p.payload["doc"],
            "page": p.payload["page"],
            "heading": p.payload.get("heading", ""),
            "text": p.payload["text"],
        }
        if rerank_scores is not None:
            item["rerank_score"] = round(rerank_scores[i], 4)
        results.append(item)
    return {"question": req.question, "reranked": use_rerank, "results": results}
