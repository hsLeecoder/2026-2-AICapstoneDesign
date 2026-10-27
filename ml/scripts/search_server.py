"""질문을 벡터로 바꾸고(M5) Qdrant에서 비슷한 청크를 찾아주는(M6) FastAPI 서버.

실행 (EC2 터미널, 프로젝트 루트에서, (pytorch) 환경 켠 상태):
    nohup python -m uvicorn --app-dir ml/scripts search_server:app --host 127.0.0.1 --port 8000 > ml/output/server.log 2>&1 &

설정은 환경변수로 바꿀 수 있다 (안 정하면 아래 기본값 사용):
    MODEL_PATH  (기본 ml/output/bge-m3-finetuned/final)
    QDRANT_URL  (기본 http://localhost:6333)
    COLLECTION  (기본 welfare_chunks_bgem3)
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

state = {}  # 서버가 켜질 때 model, client, collection 을 채운다


@asynccontextmanager
async def lifespan(app):
    import torch
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient

    model_path = os.environ.get("MODEL_PATH", "ml/output/bge-m3-finetuned/final")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    model.max_seq_length = 512
    state["model"] = model
    state["model_path"] = model_path
    state["client"] = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    state["collection"] = os.environ.get("COLLECTION", "welfare_chunks_bgem3")
    print(f"[서버 준비 완료] 모델: {model_path} / 장치: {device} / 컬렉션: {state['collection']}")
    yield
    state.clear()


app = FastAPI(title="규정 검색 서버 (M5 질의 임베딩 + M6 벡터 검색)", lifespan=lifespan)


class EmbedRequest(BaseModel):
    question: str = Field(min_length=1, description="직원의 질문")


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, description="직원의 질문")
    top_k: int = Field(5, ge=1, le=20, description="돌려줄 청크 개수")


def embed(question: str):
    return state["model"].encode([question], normalize_embeddings=True)[0].tolist()


@app.get("/health")
def health():
    count = state["client"].count(state["collection"]).count
    return {"status": "ok", "model": state["model_path"], "collection": state["collection"], "chunks": count}


@app.post("/embed")
def embed_endpoint(req: EmbedRequest):
    """M5 질의 임베딩: 질문(문자열) -> 질문 벡터"""
    vec = embed(req.question)
    return {"dim": len(vec), "vector": vec}


@app.post("/search")
def search(req: SearchRequest):
    """M5 + M6: 질문 -> 유사도 상위 k개 청크 (본문, 출처, 점수)"""
    vec = embed(req.question)
    points = state["client"].query_points(
        state["collection"], query=vec, limit=req.top_k, with_payload=True
    ).points
    return {
        "question": req.question,
        "results": [
            {
                "score": round(p.score, 4),
                "chunk_id": p.payload["chunk_id"],
                "doc": p.payload["doc"],
                "page": p.payload["page"],
                "heading": p.payload.get("heading", ""),
                "text": p.payload["text"],
            }
            for p in points
        ],
    }
