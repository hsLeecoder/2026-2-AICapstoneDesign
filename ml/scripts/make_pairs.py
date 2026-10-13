"""청크 → 학습 문제집(질문, 정답 청크 쌍) 생성 스크립트.

하는 일:
    청크(규정 조각)를 OpenAI에게 보여주고 "직원이 물어볼 법한 질문"을 만들게 한다.
    질문과 그 청크를 짝지어 학습 문제집(train.jsonl)으로 저장한다.
    (이 짝이 임베딩 모델 학습의 "문제와 정답"이 된다)

준비:
    OpenAI API 키가 필요하다. 레포 폴더의 .env 파일에 OPENAI_API_KEY=... 가 있으면 자동으로 읽는다
    (Spring 설정과 같은 파일, 같은 이름). 키는 코드, 명령어 옵션, git 에 절대 넣지 않는다.
    .env 에 OPENAI_CHAT_MODEL=모델이름 이 있으면 그 모델을 기본으로 쓴다 (--model 로 덮어쓸 수 있음).

사용 예:
    # 1) API를 부르지 않고, 첫 청크로 어떤 요청을 보낼지만 확인 (키 불필요)
    python ml/scripts/make_pairs.py --chunks ml/data/chunks/train_chunks.jsonl --dry-run

    # 2) 소량(청크 5개)으로 먼저 시험
    python ml/scripts/make_pairs.py --chunks ml/data/chunks/train_chunks.jsonl --limit 5 --output ml/data/pairs/try.jsonl

    # 3) 전체 생성 (중간에 끊겨도 같은 명령으로 다시 실행하면 이어서 한다)
    python ml/scripts/make_pairs.py --chunks ml/data/chunks/train_chunks.jsonl --output ml/data/pairs/train.jsonl

출력 한 줄의 모양 (train_embedding.py 가 읽는 형식과 같다):
    {"query": "질문", "positive": "정답이 들어 있는 규정 조각", "doc": "파일명.pdf", "page": 5, "chunk_id": "..."}

주의: 문제집에는 규정 원문이 그대로 들어 있으므로 GitHub에 올리지 않는다 (.gitignore 대상).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "gpt-5.4-mini"  # 비용이 낮은 작은 모델. 내 계정에서 쓸 수 있는 이름으로 --model 로 바꿀 수 있다.
DEFAULT_QUESTIONS = 4  # 청크 하나당 만들 질문 수
MAX_RETRIES = 3
MAX_CONSECUTIVE_FAILURES = 3  # 연속으로 이만큼 실패하면 키/모델 이름 문제로 보고 멈춘다

SYSTEM_PROMPT = """당신은 사내 규정 Q&A 챗봇의 학습 데이터를 만드는 도우미입니다.
주어진 [규정 조각]을 읽고, 직원이 챗봇에게 실제로 물어볼 법한 한국어 질문을 만드세요.

규칙:
1. 모든 질문은 반드시 이 조각 안의 내용만으로 답할 수 있어야 합니다.
2. 조항 번호나 "이 규정", "위 조항", "본문" 같은 표현을 질문에 쓰지 마세요. 규정을 읽지 않은 직원이 묻는 것처럼 쓰세요.
3. 질문끼리 서로 다른 관점(금액, 기간, 대상, 절차, 조건 등)과 말투(공손한 말투, 짧은 말투 등)로 다양하게 만드세요.
4. 규정 문장을 그대로 베끼지 말고 일상적인 표현으로 바꾸세요.
5. 규정에 없는 내용을 지어내지 마세요.

반드시 {"questions": ["질문1", "질문2"]} 형태의 JSON만 출력하세요."""


def load_env_file() -> None:
    """레포의 .env 파일이 있으면 그 안의 값을 환경변수로 불러온다. 이미 설정된 환경변수는 덮어쓰지 않는다."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return  # python-dotenv 가 없으면 직접 설정한 환경변수만 사용
    # 1) 명령을 실행한 폴더(위로 거슬러 올라가며) 2) 이 스크립트가 있는 폴더(위로 거슬러 올라가며) 순서로 .env 를 찾는다
    env_path = find_dotenv(usecwd=True) or find_dotenv()
    if env_path:
        load_dotenv(env_path)


def load_chunks(path: Path) -> list[dict]:
    """청크 파일(.jsonl)을 읽는다."""
    if not path.exists():
        raise FileNotFoundError(f"청크 파일을 찾을 수 없습니다: {path} (prepare_chunks.py 를 먼저 실행하세요)")
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name} {line_no}번째 줄이 올바른 JSON이 아닙니다: {e}") from e
            for key in ("chunk_id", "doc", "page", "text"):
                if key not in row:
                    raise ValueError(f"{path.name} {line_no}번째 줄에 '{key}' 칸이 없습니다.")
            chunks.append(row)
    return chunks


def load_done_ids(output: Path) -> set[str]:
    """이미 문제를 만든 청크 id 목록 (끊겼을 때 이어서 하기 위함)."""
    done: set[str] = set()
    if output.exists():
        with output.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line).get("chunk_id", ""))
    return done


def build_messages(text: str, n: int) -> list[dict]:
    """OpenAI에 보낼 대화 메시지를 만든다."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"질문 {n}개를 만드세요. JSON으로만 답하세요.\n\n[규정 조각]\n{text}"},
    ]


def clean_questions(raw: object, n: int) -> list[str]:
    """모델이 돌려준 질문을 정리한다: 문자열만, 너무 짧거나 긴 것 제외, 중복 제거."""
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        question = item.strip()
        key = "".join(question.split())
        if not (6 <= len(question) <= 150) or key in seen:
            continue
        seen.add(key)
        result.append(question)
    return result[:n]


def generate_questions(client, model: str, text: str, n: int) -> tuple[list[str], int, int]:
    """청크 하나에서 질문을 만든다. (질문 목록, 입력 토큰, 출력 토큰)을 돌려준다."""
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(text, n),
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return clean_questions(data.get("questions"), n), prompt_tokens, completion_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="청크에서 질문을 합성해 학습 문제집 생성")
    parser.add_argument("--chunks", required=True, help="prepare_chunks.py 가 만든 청크 .jsonl")
    parser.add_argument("--output", default="ml/data/pairs/train.jsonl", help="문제집을 저장할 .jsonl 경로")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_CHAT_MODEL") or DEFAULT_MODEL,
        help="사용할 OpenAI 모델 이름 (기본: .env 의 OPENAI_CHAT_MODEL, 없으면 코드의 기본값)",
    )
    parser.add_argument("--questions-per-chunk", type=int, default=DEFAULT_QUESTIONS, help="청크 하나당 만들 질문 수")
    parser.add_argument("--min-chars", type=int, default=40, help="이보다 짧은 청크는 건너뜀")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수의 청크만 처리 (시험용)")
    parser.add_argument("--dry-run", action="store_true", help="API를 부르지 않고 요청 내용만 출력")
    return parser.parse_args()


def main() -> None:
    load_env_file()  # parse_args 보다 먼저 해야 .env 의 모델 이름이 기본값에 반영된다
    args = parse_args()
    chunks = [c for c in load_chunks(Path(args.chunks)) if len(c["text"]) >= args.min_chars]
    if args.limit is not None:
        chunks = chunks[: args.limit]
    if not chunks:
        raise ValueError("처리할 청크가 없습니다.")

    if args.dry_run:
        print(f"[dry-run] 처리 대상 청크 {len(chunks)}개. 첫 청크로 보낼 요청 내용:\n")
        for message in build_messages(chunks[0]["text"], args.questions_per_chunk):
            print(f"--- {message['role']} ---\n{message['content']}\n")
        print("[dry-run] API를 부르지 않고 종료합니다.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[오류] OPENAI_API_KEY 를 찾을 수 없습니다. .env 파일에 OPENAI_API_KEY=... 를 적거나 환경변수로 설정한 뒤 다시 실행하세요.")

    from openai import OpenAI  # 이 단계에서만 필요하므로 여기서 불러온다

    client = OpenAI()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output)
    todo = [c for c in chunks if c["chunk_id"] not in done_ids]
    print(f"[시작] 청크 {len(chunks)}개 중 이미 처리됨 {len(chunks) - len(todo)}개, 이번에 처리할 것 {len(todo)}개 (모델: {args.model})")

    written = failed = consecutive_failures = 0
    total_in = total_out = 0
    with output.open("a", encoding="utf-8") as f:
        for index, chunk in enumerate(todo, start=1):
            questions: list[str] = []
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    questions, tokens_in, tokens_out = generate_questions(
                        client, args.model, chunk["text"], args.questions_per_chunk
                    )
                    total_in += tokens_in
                    total_out += tokens_out
                    break
                except Exception as e:  # 네트워크, 요청 한도, 응답 형식 오류 등
                    print(f"  [재시도 {attempt}/{MAX_RETRIES}] {chunk['chunk_id']}: {type(e).__name__}: {e}")
                    time.sleep(2**attempt)

            if not questions:
                failed += 1
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    sys.exit("[중단] 연속으로 실패했습니다. API 키와 모델 이름(--model)을 확인하세요.")
                continue
            consecutive_failures = 0

            for question in questions:
                row = {
                    "query": question,
                    "positive": chunk["text"],
                    "doc": chunk["doc"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            f.flush()  # 중간에 끊겨도 지금까지의 결과가 남도록 바로 기록
            if index % 10 == 0:
                print(f"  진행 {index}/{len(todo)} (문제 {written}개)")

    print(f"\n[완료] 문제 {written}개 저장: {output} / 실패한 청크 {failed}개")
    print(f"[사용량] 입력 토큰 {total_in:,} / 출력 토큰 {total_out:,}")

    # 눈으로 품질을 확인할 수 있도록 무작위 샘플을 보여준다
    if written:
        with output.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        print("\n[품질 확인용 무작위 샘플]")
        for row in random.sample(rows, min(5, len(rows))):
            print(f"  질문: {row['query']}")
            print(f"  정답 조각: {row['positive'][:70].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
