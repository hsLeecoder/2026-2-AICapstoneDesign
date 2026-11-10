"""규정 Q&A 서버: 질문 임베딩(M5) -> 벡터 검색(M6) -> 리랭킹(M7) -> 답변 생성(M8).

실행 (EC2 터미널, 프로젝트 루트에서, (pytorch) 환경 켠 상태):
    # 검색만 (가벼움)
    nohup python -m uvicorn --app-dir ml/scripts search_server:app --host 127.0.0.1 --port 8000 > ml/output/server.log 2>&1 &
    # 답변 생성(LLM)까지
    USE_LLM=1 nohup python -m uvicorn --app-dir ml/scripts search_server:app --host 127.0.0.1 --port 8000 > ml/output/server.log 2>&1 &

설정은 환경변수로 바꿀 수 있다 (안 정하면 아래 기본값 사용):
    MODEL_PATH        (기본 ml/output/bge-m3-finetuned/final)   임베딩 모델
    RERANKER_PATH     (기본 BAAI/bge-reranker-v2-m3)             리랭커 모델
    USE_RERANKER      (기본 1, 끄려면 0)
    USE_LLM           (기본 0, 답변 생성을 쓰려면 1)
    LLM_PATH          (기본 Qwen/Qwen3-8B)                       답변 생성 LLM (GPU 메모리가 모자라면 Qwen/Qwen3-4B)
    PROMPT_VERSION    (기본 v1)                                  답변 프롬프트 버전 (v1=기존, v2=근거 충실성 강화). 요청마다 prompt_version 으로 바꿀 수도 있음
    MIN_RERANK_SCORE  (기본 0.1)                                 1위 리랭커 점수가 이보다 낮으면 "확인되지 않음"으로 답함
    QDRANT_URL        (기본 http://localhost:6333)
    COLLECTION        (기본 welfare_chunks_bgem3)
"""
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

state = {}  # 서버가 켜질 때 model, reranker, llm, client, collection 을 채운다

NO_ANSWER = "규정에서 확인되지 않습니다. 질문을 조금 더 구체적으로 바꾸거나 인사 담당 부서에 문의해 주세요."

SYSTEM_PROMPT = (
    "당신은 사내 규정 안내 챗봇입니다.\n"
    "아래 [규정 근거]에 적힌 내용만으로 답하세요.\n"
    "근거는 PDF에서 추출한 글이라 규정 조문과 작성 안내문이 뒤섞여 있고 문장이 중간에 끊겨 있을 수 있습니다. "
    "조문의 항 번호(①②③)를 따라 읽으세요. '[필수]', '[선택]', '☞ (참고)', '취업규칙(안) (작성시 착안사항)', '*'로 시작하는 문장과 "
    "법령 해설, 정부 지원 안내는 규정이 아니라 작성 안내이므로 답변 근거로 쓰지 마세요.\n"
    "규칙:\n"
    "1. 질문이 묻는 내용에 먼저 직접 답하세요. 묻지 않은 조건이나 부가 설명은 쓰지 마세요.\n"
    "2. 기간·일수·금액·비율 같은 숫자와 조건은 근거 문장 그대로 정확히 쓰세요. 일부만 옮기거나 바꾸지 마세요.\n"
    "3. 근거에 없는 내용은 추측하거나 지어내지 마세요. 근거만으로 답할 수 없으면 '규정에서 확인되지 않습니다.'라고만 답하세요.\n"
    "4. 한국어로 간결하게(최대 5문장) 답하세요.\n"
    "5. 답변 맨 마지막 줄에 실제로 사용한 근거 번호를 [근거: 1, 2] 형식으로 적으세요."
)


# v2: 근거 충실성을 높인 프롬프트. v1 과 달리 (1) 답에 영향을 주는 조건·예외를 빠뜨리지 않게 하고,
# (2) 질문 대상이 근거에 없으면 비슷한 다른 제도를 적용하지 못하게 하며, (3) 단정적인 질문의 전제를 근거대로 바로잡고,
# (4) 문서가 섞이면 출처 문서를 구분하게 한다.
SYSTEM_PROMPT_V2 = (
    "당신은 사내 규정 안내 챗봇입니다.\n"
    "아래 [규정 근거]에 적힌 내용만으로 답하세요.\n"
    "근거는 PDF에서 추출한 글이라 규정 조문과 작성 안내문이 뒤섞여 있고 문장이 중간에 끊겨 있을 수 있습니다. "
    "조문의 항 번호(①②③)를 따라 읽으세요. '[필수]', '[선택]', '☞ (참고)', '취업규칙(안) (작성시 착안사항)', '*'로 시작하는 문장과 "
    "법령 해설, 정부 지원 안내는 규정이 아니라 작성 안내이므로 답변 근거로 쓰지 마세요.\n"
    "규칙:\n"
    "1. 질문이 묻는 내용에 먼저 직접 답하세요. 그 답에 영향을 주는 조건·예외·가산 기준·상한·횟수가 근거에 있으면 빠뜨리지 말고 함께 쓰세요. 질문과 무관한 내용은 쓰지 마세요.\n"
    "2. 기간·일수·금액·비율 같은 숫자와 조건은 근거 문장 그대로 정확히 쓰세요. 일부만 옮기거나 바꾸지 마세요.\n"
    "3. 질문이 가리키는 제도·용어(예: 반차, 시차, 특정 수당)가 근거에 직접 나오지 않으면, 비슷해 보이는 다른 제도의 규정을 그 대상에 적용하지 마세요. "
    "근거에 직접 적혀 있지 않은 지급·허용·신청 기한 여부를 단정하지 마세요.\n"
    "4. 질문의 일부만 근거로 답할 수 있으면 확인되는 부분만 답하고, 나머지는 '규정에 명시되어 있지 않습니다'라고 쓰세요. "
    "질문 전체가 근거만으로 답할 수 없을 때만 '규정에서 확인되지 않습니다.'라고만 답하세요.\n"
    "5. 질문이 '무조건', '반드시'처럼 단정적으로 묻더라도 근거가 그렇지 않으면 근거대로 바로잡아 답하세요. 근거에 예외가 있으면 그 예외를 함께 쓰세요.\n"
    "6. 근거가 서로 다른 문서에서 왔으면 어느 문서의 내용인지 구분해서 쓰세요.\n"
    "7. 한국어로 간결하게(최대 6문장) 답하세요.\n"
    "8. 답변 맨 마지막 줄에 실제로 사용한 근거 번호를 [근거: 1, 2] 형식으로 적으세요."
)

PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2}

# v2 에서 근거 머리말에 쓰는 문서 이름표 (모르는 문서는 파일 이름 그대로 표시)
DOC_LABELS = {
    "welfare_ananti.pdf": "복리후생규정(아난티)",
    "workrules_goyongnodong.pdf": "취업규칙(고용노동부 표준)",
}


@asynccontextmanager
async def lifespan(app):
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from qdrant_client import QdrantClient

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.environ.get("MODEL_PATH", "ml/output/bge-m3-finetuned/final")
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

    state["llm"] = None
    state["llm_path"] = None
    if os.environ.get("USE_LLM", "0") == "1":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        llm_path = os.environ.get("LLM_PATH", "Qwen/Qwen3-8B")
        state["tokenizer"] = AutoTokenizer.from_pretrained(llm_path)
        # device_map="cuda"로 GPU에 바로 올린다 (서버 RAM이 16GB라 CPU를 거치면 모자랄 수 있음)
        state["llm"] = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch.bfloat16, device_map="cuda")
        state["llm_path"] = llm_path

    state["prompt_version"] = os.environ.get("PROMPT_VERSION", "v1")
    if state["prompt_version"] not in PROMPTS:
        raise ValueError(f"PROMPT_VERSION 은 {list(PROMPTS)} 중 하나여야 합니다.")
    state["min_rerank_score"] = float(os.environ.get("MIN_RERANK_SCORE", "0.1"))
    state["client"] = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    state["collection"] = os.environ.get("COLLECTION", "welfare_chunks_bgem3")
    print(f"[서버 준비 완료] 임베딩: {model_path} / 리랭커: {state['reranker_path']} / LLM: {state['llm_path']} / 장치: {device} / 컬렉션: {state['collection']} / 프롬프트: {state['prompt_version']}")
    yield
    state.clear()


app = FastAPI(title="규정 Q&A 서버 (M5~M8)", lifespan=lifespan)


class EmbedRequest(BaseModel):
    question: str = Field(min_length=1, description="직원의 질문")


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, description="직원의 질문")
    top_k: int = Field(5, ge=1, le=20, description="돌려줄 청크 개수")
    rerank: bool = Field(True, description="리랭커로 재정렬할지 (서버에 리랭커가 없으면 무시)")
    candidates: int = Field(20, ge=1, le=50, description="리랭커에 넘길 후보 개수")


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, description="직원의 질문")
    top_k: int = Field(3, ge=1, le=6, description="답변 근거로 LLM에 넘길 청크 개수")
    prompt_version: str | None = Field(None, description="프롬프트 버전(v1/v2). 안 정하면 서버 기본값(PROMPT_VERSION)")


def embed(question: str):
    return state["model"].encode([question], normalize_embeddings=True)[0].tolist()


def retrieve(question: str, top_k: int, rerank: bool, candidates: int):
    """M5 + M6 (+ M7): 질문 -> 상위 top_k개 청크(dict 목록)와 리랭크 사용 여부"""
    use_rerank = rerank and state.get("reranker") is not None
    limit = max(top_k, candidates) if use_rerank else top_k
    points = state["client"].query_points(
        state["collection"], query=embed(question), limit=limit, with_payload=True
    ).points

    rerank_scores = None
    if use_rerank:
        pairs = [(question, p.payload["text"]) for p in points]
        rerank_scores = [float(s) for s in state["reranker"].predict(pairs, show_progress_bar=False)]
        order = sorted(range(len(points)), key=lambda i: -rerank_scores[i])[:top_k]
    else:
        order = list(range(len(points)))[:top_k]

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
    return results, use_rerank


def build_messages(question: str, results: list, version: str = "v1"):
    """M8 프롬프트: 규정 근거와 질문을 LLM 대화 형식으로 만든다"""
    if version == "v2":
        blocks = [f"[근거 {n}] ({DOC_LABELS.get(r['doc'], r['doc'])} p.{r['page']})\n{r['text']}" for n, r in enumerate(results, start=1)]
    else:
        blocks = [f"[근거 {n}] ({r['doc']} p.{r['page']})\n{r['text']}" for n, r in enumerate(results, start=1)]
    user = "[규정 근거]\n" + "\n\n".join(blocks) + f"\n\n[질문]\n{question}"
    return [{"role": "system", "content": PROMPTS[version]}, {"role": "user", "content": user}]


def generate_text(messages: list) -> str:
    """LLM으로 답변 생성 (결정적 생성: 같은 입력이면 같은 답)"""
    tok, llm = state["tokenizer"], state["llm"]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(**inputs, max_new_tokens=400, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def split_citation(text: str, n_sources: int):
    """답변 끝의 [근거: 1, 2]를 떼어내 근거 번호 목록(0부터)으로 바꾼다. 없으면 전부 사용한 것으로 본다."""
    m = re.search(r"\[근거\s*:\s*([\d,\s]+)\]\s*$", text)
    if not m:
        return text.strip(), list(range(n_sources))
    nums = [int(x) - 1 for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= n_sources]
    return text[: m.start()].strip(), (sorted(set(nums)) or list(range(n_sources)))


@app.get("/health")
def health():
    count = state["client"].count(state["collection"]).count
    return {"status": "ok", "model": state["model_path"], "reranker": state["reranker_path"],
            "llm": state["llm_path"], "collection": state["collection"], "chunks": count}


@app.post("/embed")
def embed_endpoint(req: EmbedRequest):
    """M5 질의 임베딩: 질문(문자열) -> 질문 벡터"""
    vec = embed(req.question)
    return {"dim": len(vec), "vector": vec}


@app.post("/search")
def search(req: SearchRequest):
    """M5 + M6 (+ M7): 질문 -> 상위 k개 청크 (본문, 출처, 점수). 리랭커를 쓰면 rerank_score도 함께 반환"""
    results, used = retrieve(req.question, req.top_k, req.rerank, req.candidates)
    return {"question": req.question, "reranked": used, "results": results}


@app.post("/answer")
def answer(req: AnswerRequest):
    """M5~M8: 질문 -> 검색·재정렬 -> (근거 충분하면) LLM 답변 + 출처. 근거가 약하면 LLM 없이 '확인되지 않음'"""
    if state.get("llm") is None:
        return {"error": "LLM이 꺼져 있습니다. USE_LLM=1 로 서버를 다시 켜세요."}
    version = req.prompt_version or state["prompt_version"]
    if version not in PROMPTS:
        raise HTTPException(status_code=400, detail=f"prompt_version 은 {list(PROMPTS)} 중 하나여야 합니다.")
    results, used = retrieve(req.question, req.top_k, True, 20)
    top_score = results[0].get("rerank_score") if results else None

    # 리랭커 1위 점수가 기준보다 낮으면 규정에 없는 질문으로 보고 LLM을 부르지 않는다
    if not results or (used and top_score < state["min_rerank_score"]):
        return {"question": req.question, "answered": False, "answer": NO_ANSWER, "sources": [], "top_score": top_score, "prompt_version": version}

    text = generate_text(build_messages(req.question, results, version))
    body, used_idx = split_citation(text, len(results))
    answered = "확인되지 않습니다" not in body
    sources = [
        {"doc": results[i]["doc"], "page": results[i]["page"], "heading": results[i]["heading"],
         "rerank_score": results[i].get("rerank_score")}
        for i in used_idx
    ] if answered else []
    return {"question": req.question, "answered": answered, "answer": body, "sources": sources, "top_score": top_score, "prompt_version": version}
