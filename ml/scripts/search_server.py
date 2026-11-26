"""규정 Q&A 서버: 질문 임베딩(M5) -> 벡터 검색(M6) -> 리랭킹(M7) -> 답변 생성(M8).

실행 (EC2 터미널, 프로젝트 루트에서, (pytorch) 환경 켠 상태):
    # 검색만 (가벼움)
    nohup python -m uvicorn --app-dir ml/scripts search_server:app --host 127.0.0.1 --port 8000 > ml/output/server.log 2>&1 &
    # 답변 생성(LLM)까지
    USE_LLM=1 nohup python -m uvicorn --app-dir ml/scripts search_server:app --host 127.0.0.1 --port 8000 > ml/output/server.log 2>&1 &

설정은 환경변수로 바꿀 수 있다 (안 정하면 아래 기본값 사용):
    MODEL_PATH        (기본 ml/output/bge-m3-finetuned-v3/final) 임베딩 모델 (3차 모델)
    RERANKER_PATH     (기본 BAAI/bge-reranker-v2-m3)             리랭커 모델
    USE_RERANKER      (기본 1, 끄려면 0)
    USE_LLM           (기본 0, 답변 생성을 쓰려면 1)
    LLM_PATH          (기본 Qwen/Qwen3-8B)                       답변 생성 LLM (GPU 메모리가 모자라면 Qwen/Qwen3-4B)
    PROMPT_VERSION    (기본 v4)                                  답변 프롬프트 버전 (v1=기존, v2=근거 충실성 강화, v3=v2+말투·질문 대상 일치, v4=v3+숫자 조건 보존, v5=v4+전제 수용·없는 용어 적용 방지+재작성 개선+나열형 근거 5개). 요청마다 prompt_version 으로 바꿀 수도 있음
    VERIFY_NUMBERS    (기본 0, 켜려면 1)                         답변이 근거의 숫자 조건을 빠뜨리면 한 번 다시 생성. 평가에서 점수 개선이 없고 응답이 느려져 기본은 꺼 둠. 요청마다 verify_numbers 로 바꿀 수도 있음
    MIN_RERANK_SCORE  (기본 0.005)                               1위 리랭커 점수가 이보다 낮으면 "확인되지 않음"으로 답함
    MAX_HISTORY_TURNS (기본 3)                                   멀티턴: 질문 재작성에 쓰는 최근 대화 턴 수
    EXCLUDE_DOCS      (기본 비어 있음)                           검색에서 뺄 문서 파일명(쉼표로 여러 개). 예: welfare_ananti.pdf. 요청마다 exclude_docs 로 바꿀 수도 있음(빈 목록이면 제외 없음)
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

# v3: v2 에 말투와 질문 대상 일치 규칙을 더했다. (1) 존댓말로 일관되게, 조문 말투를 풀어 쓰기,
# (2) 예/아니오 질문이 아니면 '네,'로 시작하지 않기, (3) 근거가 다루는 제도가 질문이 묻는 제도와 같은지 먼저 확인하기.
SYSTEM_PROMPT_V3 = SYSTEM_PROMPT_V2.replace(
    "7. 한국어로 간결하게(최대 6문장) 답하세요.\n"
    "8. 답변 맨 마지막 줄에 실제로 사용한 근거 번호를 [근거: 1, 2] 형식으로 적으세요.",
    "7. 근거가 다루는 제도(예: 연차, 병가, 경조사휴가)와 질문이 묻는 제도가 같은지 먼저 확인하세요. 다르면 근거의 내용을 질문 대상에 적용하지 말고 '규정에 명시되어 있지 않습니다'라고 쓰세요.\n"
    "8. 항상 존댓말(~합니다, ~입니다, ~해요)로 답하세요. 근거의 '~한다', '~준다' 같은 조문 말투와 '제1항 규정에 따른', '계속 근로연수' 같은 표현을 그대로 옮기지 말고 직원이 이해하기 쉬운 말로 풀어 쓰세요. 단 숫자와 조건은 그대로 유지하세요.\n"
    "9. 질문이 예/아니오를 묻는 형태가 아니면 '네,'로 시작하지 마세요.\n"
    "10. 한국어로 간결하게(최대 6문장) 답하세요.\n"
    "11. 답변 맨 마지막 줄에 실제로 사용한 근거 번호를 [근거: 1, 2] 형식으로 적으세요.",
)
assert SYSTEM_PROMPT_V3 != SYSTEM_PROMPT_V2

# v4: v3 에서 조건(숫자에 붙은 수식어)이 사라지는 문제를 막는다.
# v3 의 "쉬운 말로 풀어 쓰기"와 "간결하게"가 "조건을 그대로 쓰기"를 눌러서
# '매 2년에 대하여 1일 가산' 같은 주기 조건이 답변에서 빠지는 일이 있었다.
# 그래서 (1) 숫자에 붙은 수식어를 반드시 남기게 하고, (2) 가산 규정은 '얼마나/얼마마다/상한'을
# 함께 쓰도록 좋은 예·나쁜 예를 주고, (3) 조건을 쓸 자리를 위해 문장 수 제한을 7문장으로 늘렸다.
SYSTEM_PROMPT_V4 = SYSTEM_PROMPT_V3.replace(
    "2. 기간·일수·금액·비율 같은 숫자와 조건은 근거 문장 그대로 정확히 쓰세요. 일부만 옮기거나 바꾸지 마세요.\n",
    "2. 기간·일수·금액·비율 같은 숫자와 조건은 근거 문장 그대로 정확히 쓰세요. 일부만 옮기거나 바꾸지 마세요.\n"
    "2-1. 숫자에 붙은 수식어('매 2년에 대하여', '3년 이상', '1개월 개근 시', '80퍼센트 이상 출근', '25일을 한도로')는 하나도 빼지 마세요. "
    "쉬운 말로 풀어 쓸 때도 이 수식어는 반드시 남기세요.\n"
    "2-2. 가산·추가·증가·한도 규정은 (1) 얼마나 늘어나는지 (2) 무엇마다(몇 년마다) 늘어나는지 (3) 상한이 얼마인지를 모두 함께 쓰세요. "
    "셋 중 근거에 적혀 있는 것은 하나도 빠뜨리면 안 됩니다.\n"
    "   나쁜 예: '3년 이상 근속하면 1일씩 가산되어 최대 25일입니다.' → '매 2년마다'가 빠졌습니다.\n"
    "   좋은 예: '3년 이상 근속하면 최초 1년을 넘는 근속기간 매 2년마다 1일이 더해지고, 가산을 포함해 최대 25일까지입니다.'\n",
).replace(
    "10. 한국어로 간결하게(최대 6문장) 답하세요.\n",
    "10. 한국어로 간결하게(최대 7문장) 답하세요. 단 2-1·2-2의 조건을 모두 쓰기 위해 필요하면 문장을 더 써도 됩니다.\n",
)
assert SYSTEM_PROMPT_V4 != SYSTEM_PROMPT_V3 and "매 2년마다" in SYSTEM_PROMPT_V4
assert "최대 7문장" in SYSTEM_PROMPT_V4

# v5: v4 에서 남은 실패 두 가지를 막는다 (v4 평가에서 만점이 아니었던 질문들의 공통 원인).
# (1) 질문의 전제를 근거 확인 없이 받아들임 ("퇴직금 받으려면 무조건 IRP 만들어야 함?" → "네, 반드시 만들어야 합니다")
# (2) 규정에 없는 제도·용어에 비슷한 다른 조항을 갖다 붙임 ("시차 신청은 며칠 전?" → 다른 휴가의 신청 기한을 시차에 적용)
# 또 v4 는 근거 머리말에 문서 이름표가 빠져 있었는데(평가된 그대로 둠), v5 는 이름표를 다시 붙인다.
SYSTEM_PROMPT_V5 = SYSTEM_PROMPT_V4.replace(
    "4. 질문의 일부만 근거로 답할 수 있으면",
    "3-1. 질문에 나온 제도·용어(특정 휴가 이름, 수당 이름, 신청 방식 등)가 [규정 근거]에 그대로 나오는지 먼저 확인하세요. "
    "나오지 않으면 '질문하신 OO에 대한 규정은 확인되지 않습니다.'로 답을 시작하고, 비슷해 보이는 다른 제도의 내용을 그 대상의 규정인 것처럼 쓰지 마세요. "
    "필요하면 '참고로' 뒤에 근거에 실제로 있는 관련 내용을 한 문장만 덧붙이세요.\n"
    "4. 질문의 일부만 근거로 답할 수 있으면",
).replace(
    "6. 근거가 서로 다른 문서에서 왔으면",
    "5-1. 질문이 어떤 사실을 전제하거나 유도하면('~도 늘어나나요?', '~해야 하나요?', '무조건 ~인가요?') '네'라고 답하기 전에 근거에 그 내용이 직접 적혀 있는지 확인하세요. "
    "직접 적혀 있지 않거나 근거가 다르면 '아니요' 또는 '규정에는 그런 내용이 없습니다'로 시작하고, 근거에 실제로 적힌 내용(조건·한도)을 이어서 쓰세요. 질문의 전제를 그대로 받아 적지 마세요.\n"
    "6. 근거가 서로 다른 문서에서 왔으면",
)
assert "3-1." in SYSTEM_PROMPT_V5 and "5-1." in SYSTEM_PROMPT_V5 and SYSTEM_PROMPT_V5 != SYSTEM_PROMPT_V4

PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3, "v4": SYSTEM_PROMPT_V4, "v5": SYSTEM_PROMPT_V5}

# v2 에서 근거 머리말에 쓰는 문서 이름표 (모르는 문서는 파일 이름 그대로 표시)
DOC_LABELS = {
    "welfare_ananti.pdf": "복리후생규정(아난티)",
    "workrules_goyongnodong.pdf": "취업규칙(고용노동부 표준)",
    "welfare_goyongnodong.pdf": "선진기업복지제도 안내(노동부·근로복지공단)",
}


@asynccontextmanager
async def lifespan(app):
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from qdrant_client import QdrantClient

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.environ.get("MODEL_PATH", "ml/output/bge-m3-finetuned-v3/final")
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

    state["exclude_docs"] = [d.strip() for d in os.environ.get("EXCLUDE_DOCS", "").split(",") if d.strip()]
    state["max_history_turns"] = int(os.environ.get("MAX_HISTORY_TURNS", "3"))
    state["verify_numbers"] = os.environ.get("VERIFY_NUMBERS", "0") == "1"
    state["prompt_version"] = os.environ.get("PROMPT_VERSION", "v4")
    if state["prompt_version"] not in PROMPTS:
        raise ValueError(f"PROMPT_VERSION 은 {list(PROMPTS)} 중 하나여야 합니다.")
    state["min_rerank_score"] = float(os.environ.get("MIN_RERANK_SCORE", "0.005"))
    state["client"] = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    state["collection"] = os.environ.get("COLLECTION", "welfare_chunks_bgem3")
    print(f"[서버 준비 완료] 임베딩: {model_path} / 리랭커: {state['reranker_path']} / LLM: {state['llm_path']} / 장치: {device} / 컬렉션: {state['collection']} / 프롬프트: {state['prompt_version']} / 숫자 검증: {'켜짐' if state['verify_numbers'] else '꺼짐'} / 제외 문서: {state['exclude_docs'] or '없음'}")
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
    exclude_docs: list[str] | None = Field(None, description="검색에서 뺄 문서(안 정하면 서버 기본값, 빈 목록이면 제외 없음)")


class HistoryTurn(BaseModel):
    question: str = Field("", description="이전 질문")
    answer: str = Field("", description="이전 답변")


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, description="직원의 질문")
    top_k: int = Field(3, ge=1, le=6, description="답변 근거로 LLM에 넘길 청크 개수")
    prompt_version: str | None = Field(None, description="프롬프트 버전(v1~v5). 안 정하면 서버 기본값(PROMPT_VERSION)")
    verify_numbers: bool | None = Field(None, description="숫자 조건 누락 검증 후 1회 재생성. 안 정하면 서버 기본값(VERIFY_NUMBERS)")
    exclude_docs: list[str] | None = Field(None, description="검색에서 뺄 문서(안 정하면 서버 기본값, 빈 목록이면 제외 없음)")
    history: list[HistoryTurn] | None = Field(None, description="멀티턴: 이전 대화(오래된 것부터). 있으면 질문을 독립된 질문으로 재작성한 뒤 검색")


def embed(question: str):
    return state["model"].encode([question], normalize_embeddings=True)[0].tolist()


def retrieve(question: str, top_k: int, rerank: bool, candidates: int, exclude: list | None = None):
    """M5 + M6 (+ M7): 질문 -> 상위 top_k개 청크(dict 목록)와 리랭크 사용 여부.
    exclude 가 None 이면 서버 기본값(EXCLUDE_DOCS)을 쓰고, 빈 목록이면 아무 문서도 빼지 않는다."""
    use_rerank = rerank and state.get("reranker") is not None
    limit = max(top_k, candidates) if use_rerank else top_k
    exclude = state.get("exclude_docs") if exclude is None else exclude
    kwargs = {}
    if exclude:
        from qdrant_client import models
        kwargs["query_filter"] = models.Filter(
            must_not=[models.FieldCondition(key="doc", match=models.MatchAny(any=exclude))]
        )
    points = state["client"].query_points(
        state["collection"], query=embed(question), limit=limit, with_payload=True, **kwargs
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
    if version in ("v2", "v3", "v5"):   # v4 는 평가 당시 이름표 없이 동작해서 그대로 둔다
        blocks = [f"[근거 {n}] ({DOC_LABELS.get(r['doc'], r['doc'])} p.{r['page']})\n{r['text']}" for n, r in enumerate(results, start=1)]
    else:
        blocks = [f"[근거 {n}] ({r['doc']} p.{r['page']})\n{r['text']}" for n, r in enumerate(results, start=1)]
    user = "[규정 근거]\n" + "\n\n".join(blocks) + f"\n\n[질문]\n{question}"
    return [{"role": "system", "content": PROMPTS[version]}, {"role": "user", "content": user}]


def generate_text(messages: list, max_new_tokens: int = 400) -> str:
    """LLM으로 답변 생성 (결정적 생성: 같은 입력이면 같은 답)"""
    tok, llm = state["tokenizer"], state["llm"]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


REWRITE_PROMPT = (
    "당신은 대화형 사내 규정 챗봇의 질문 정리기입니다.\n"
    "[대화 기록]을 보고 [마지막 질문]을 대화 기록 없이도 이해되는 독립된 한 문장 질문으로 바꿔 쓰세요.\n"
    "규칙:\n"
    "1. '그럼', '그건', '그 경우'처럼 생략되거나 대명사로 가리킨 대상은 대화 기록에서 찾아 채우세요.\n"
    "2. 마지막 질문이 이미 독립적이거나 새로운 주제이면 그대로 쓰세요. 이전 대화의 주제를 억지로 붙이지 마세요.\n"
    "3. 마지막 질문에 없는 사실이나 조건을 지어내지 마세요.\n"
    "4. 바꾼 질문 한 문장만 출력하세요. 설명이나 따옴표는 쓰지 마세요."
)


REWRITE_PROMPT_V3 = REWRITE_PROMPT.replace(
    "2. 마지막 질문이 이미 독립적이거나",
    "2. 생략된 대상은 가장 최근 질문과 답변에서 먼저 찾으세요. 오래된 대화보다 바로 앞 대화가 더 중요합니다.\n"
    "   마지막 질문이 이미 독립적이거나",
)
assert REWRITE_PROMPT_V3 != REWRITE_PROMPT
# v5: 재작성이 (1) 직전 질문의 대상을 놓치거나('그동안 월급은?' → '월급은 어떻게 지급되나요?')
# (2) 질문의 뜻을 비트는 문제('계정을 안 정하고 나가면요?' → '지정하지 않은 계정으로 받을 수 있나요?')를 막는다.
REWRITE_PROMPT_V5 = REWRITE_PROMPT_V3.replace(
    "4. 바꾼 질문 한 문장만 출력하세요.",
    "4. '그동안', '그때', '그 경우', '그 계정', '그건'처럼 직전 내용을 가리키는 말은 직전 질문의 구체적인 대상(제도·기간·계정 등)으로 바꿔 쓰세요. "
    "대상을 빼고 일반적인 질문으로 바꾸면 안 됩니다.\n"
    "5. 질문의 뜻과 유형을 바꾸지 마세요. '~하면요?'는 '~하면 어떻게 되나요?'로 바꾸고, 예/아니오·방법·금액 같은 질문 유형을 그대로 유지하세요. 사용자가 말하지 않은 해석을 덧붙이지 마세요.\n"
    "예시 (다른 주제):\n"
    "- 직전 질문 '지각하면 어떻게 되나요?' / 마지막 질문 '그럼 세 번이면요?' → 지각을 세 번 하면 어떻게 되나요?\n"
    "- 직전 질문 '교육훈련 기간은 얼마나 되나요?' / 마지막 질문 '그동안 월급은요?' → 교육훈련 기간 동안 월급은 지급되나요?\n"
    "6. 바꾼 질문 한 문장만 출력하세요.",
)
assert "그동안" in REWRITE_PROMPT_V5 and REWRITE_PROMPT_V5 != REWRITE_PROMPT_V3
REWRITE_PROMPTS = {"v1": REWRITE_PROMPT, "v2": REWRITE_PROMPT, "v3": REWRITE_PROMPT_V3, "v4": REWRITE_PROMPT_V3, "v5": REWRITE_PROMPT_V5}


def rewrite_question(question: str, history: list, version: str = "v2") -> str:
    """멀티턴: 이전 대화를 보고 마지막 질문을 독립된 질문으로 바꾼다. 결과가 이상하면 원래 질문을 쓴다."""
    convo = "\n".join(f"Q: {t.question.strip()}\nA: {t.answer.strip()[:300]}" for t in history)
    user = f"[대화 기록]\n{convo}\n\n[마지막 질문]\n{question}\n\n[독립된 질문]"
    raw = generate_text([{"role": "system", "content": REWRITE_PROMPTS[version]}, {"role": "user", "content": user}], max_new_tokens=100)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out = re.sub(r"^(독립된 질문|질문)\s*[:：]\s*", "", lines[0]).strip().strip('"\'“”‘’') if lines else ""
    return out if 0 < len(out) <= 200 else question


def clean_text(text: str) -> str:
    """화면에 보여줄 조항 원문: 공백·줄바꿈을 정리하고 ①②③ 항 앞에서만 줄을 나눈다."""
    t = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s*([①-⑩])", r"\n\1", t).strip()


# ---- 숫자 조건 누락 검증(C) ----------------------------------------------
# 프롬프트만으로는 '매 2년에 대하여' 같은 조건이 빠지는 것을 완전히 막지 못했다.
# 그래서 답변을 만든 뒤 규칙으로 한 번 더 확인한다:
#   1) 근거 청크를 문장 단위로 쪼갠다
#   2) 답변이 쓴 숫자가 들어 있는 문장을 '답변이 실제로 쓴 근거 문장'으로 본다
#   3) 그 문장 안에 있는데 답변에는 없는 숫자 표현을 '누락'으로 잡는다
#   4) 누락이 있으면 그 목록을 알려주고 딱 한 번 다시 생성한다
NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:개년|년|개월|주일|주|일|시간|분|회|차|명|배|퍼센트|%|만원|천원|원|개)")

def num_tokens(text: str) -> set:
    """숫자+단위 표현만 뽑는다. '퍼센트'는 '%'로, 공백은 지워서 '2 년'과 '2년'을 같게 본다."""
    return set(NUM_RE.findall((text or "").replace("퍼센트", "%").replace(" ", "")))

def split_sentences(text: str) -> list:
    """근거 청크를 문장으로 쪼갠다. 조문은 ①②③ 항 단위가 한 문장 역할을 한다."""
    t = re.sub(r"\s+", " ", text or "")
    parts = re.split(r"(?=[①-⑩])|(?<=[.])\s", t)
    return [x.strip() for x in parts if x.strip()]

def missing_numbers(answer: str, results: list, used_idx: list, limit: int = 5) -> list:
    """답변이 근거의 숫자 조건을 빠뜨렸으면 그 표현들을 돌려준다."""
    ans = num_tokens(answer)
    if not ans:
        return []
    missing = []
    for i in used_idx:
        if i >= len(results):
            continue
        for sent in split_sentences(results[i]["text"]):
            toks = num_tokens(sent)
            if not toks or not (toks & ans):   # 답변이 이 문장을 쓰지 않았다고 보고 넘어간다
                continue
            for t in toks - ans:
                if t not in missing:
                    missing.append(t)
    return missing[:limit]

REPAIR_HINT = (
    "방금 답변에 근거의 숫자 조건이 빠졌습니다. 빠진 표현: {missing}\n"
    "이 표현이 [규정 근거]의 어느 문장에 있는지 찾아보고, 그 조건을 포함해 답변을 다시 쓰세요.\n"
    "- 근거에 없는 내용은 새로 더하지 마세요.\n"
    "- 질문과 관계없는 조건이면 넣지 않아도 됩니다.\n"
    "- 마지막 줄에 [근거: 1, 2] 형식을 다시 적으세요."
)


# 나열형 질문('종류', '뭐가 있나요')은 근거 3개로는 항목을 다 못 담아서 v5 에서는 근거를 5개까지 쓴다.
LIST_Q = re.compile(r"종류|뭐가\s*있|무엇이\s*있|어떤\s*\S{0,6}\s*(?:이|가)?\s*있")


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
    results, used = retrieve(req.question, req.top_k, req.rerank, req.candidates, exclude=req.exclude_docs)
    return {"question": req.question, "reranked": used, "results": results}


@app.post("/answer")
def answer(req: AnswerRequest):
    """M5~M8: 질문 -> (멀티턴이면 질문 재작성) -> 검색·재정렬 -> (근거 충분하면) LLM 답변 + 출처. 근거가 약하면 LLM 없이 '확인되지 않음'"""
    if state.get("llm") is None:
        return {"error": "LLM이 꺼져 있습니다. USE_LLM=1 로 서버를 다시 켜세요."}
    version = req.prompt_version or state["prompt_version"]
    if version not in PROMPTS:
        raise HTTPException(status_code=400, detail=f"prompt_version 은 {list(PROMPTS)} 중 하나여야 합니다.")

    history = [t for t in (req.history or []) if t.question.strip()][-state.get("max_history_turns", 3):]
    rewritten = rewrite_question(req.question, history, version) if history else None
    question = rewritten or req.question

    verify = state.get("verify_numbers") if req.verify_numbers is None else req.verify_numbers
    top_k = max(req.top_k, 5) if (version == "v5" and LIST_Q.search(question)) else req.top_k
    results, used = retrieve(question, top_k, True, 20, exclude=req.exclude_docs)
    top_score = results[0].get("rerank_score") if results else None
    base = {"question": req.question, "rewritten_question": rewritten, "top_score": top_score,
            "prompt_version": version, "verify_numbers": verify, "top_k_used": top_k}

    # 리랭커 1위 점수가 기준보다 낮으면 규정에 없는 질문으로 보고 LLM을 부르지 않는다
    if not results or (used and top_score < state["min_rerank_score"]):
        return {**base, "answered": False, "answer": NO_ANSWER, "sources": []}

    messages = build_messages(question, results, version)
    text = generate_text(messages)
    body, used_idx = split_citation(text, len(results))
    answered = "확인되지 않습니다" not in body

    # 숫자 조건이 빠졌으면 한 번만 다시 생성한다 (더 좋아졌을 때만 바꿔 쓴다)
    check = None
    if verify and answered:
        miss = missing_numbers(body, results, used_idx)
        check = {"missing": miss, "repaired": False}
        if miss:
            retry = generate_text(messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": REPAIR_HINT.format(missing=", ".join(miss))},
            ])
            r_body, r_idx = split_citation(retry, len(results))
            if "확인되지 않습니다" not in r_body and len(missing_numbers(r_body, results, r_idx)) < len(miss):
                body, used_idx = r_body, r_idx
                check = {"missing": miss, "repaired": True, "still_missing": missing_numbers(body, results, used_idx)}

    sources = [
        {"doc": results[i]["doc"], "page": results[i]["page"], "heading": results[i]["heading"],
         "rerank_score": results[i].get("rerank_score"), "text": clean_text(results[i]["text"])}
        for i in used_idx
    ] if answered else []
    return {**base, "answered": answered, "answer": body, "sources": sources, "verify": check}
