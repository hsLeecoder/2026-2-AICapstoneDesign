"""임베딩 모델 파인튜닝 스크립트 (뼈대).

하는 일:
    (질문, 정답 청크) 쌍으로 이미 학습된 임베딩 모델을 추가 학습(파인튜닝)시킨다.
    목표: "질문"과 "그 질문의 답이 들어 있는 규정 조각"의 벡터가 서로 가까워지게 만든다.

사용 예:
    # 1) 데이터 점검만 (GPU, 모델 다운로드 없이 가볍게 실행 가능)
    python ml/scripts/train_embedding.py --train-file ml/data/sample_train.jsonl --dry-run

    # 2) 실제 학습 (EC2 GPU 서버에서)
    python ml/scripts/train_embedding.py --train-file ml/data/train.jsonl --output-dir ml/output/embedding-v1

    # 3) 연결 끊김에 대비한 백그라운드 실행
    nohup python ml/scripts/train_embedding.py --train-file ml/data/train.jsonl > train.log 2>&1 &

학습 데이터 형식 (한 줄에 JSON 하나, 파일 확장자는 .jsonl):
    {"query": "질문", "positive": "정답이 들어 있는 규정 조각", "doc": "파일명.pdf", "page": 5}
    - query, positive : 학습에 반드시 필요 (비어 있으면 에러)
    - doc, page       : 출처 정보 (학습에는 쓰지 않고 나중에 답변에 출처를 붙일 때 사용)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 기본 설정값 (바꾸고 싶을 때는 실행할 때 옵션으로 덮어쓸 수 있다)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "BAAI/bge-m3"  # 출발점이 되는 사전학습 임베딩 모델 (다국어, 한국어 지원)
REQUIRED_KEYS = ("query", "positive")  # 학습에 꼭 필요한 칸


def load_pairs(path: Path) -> list[dict]:
    """jsonl 파일을 읽어서 (질문, 정답) 목록으로 돌려준다. 형식이 틀리면 몇 번째 줄인지 알려준다."""
    if not path.exists():
        raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {path}")

    pairs: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # 빈 줄은 무시
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name} {line_no}번째 줄이 올바른 JSON이 아닙니다: {e}") from e

            for key in REQUIRED_KEYS:
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"{path.name} {line_no}번째 줄에 '{key}' 칸이 없거나 비어 있습니다.")
            pairs.append(row)

    if not pairs:
        raise ValueError(f"{path.name} 에 학습 데이터가 한 줄도 없습니다.")
    return pairs


def summarize(pairs: list[dict]) -> None:
    """데이터가 어떻게 생겼는지 요약해서 보여준다 (학습 전 눈으로 확인하는 용도)."""
    queries = [p["query"] for p in pairs]
    positives = [p["positive"] for p in pairs]
    unique_positives = len(set(positives))

    print(f"[데이터] 총 {len(pairs)}건 / 서로 다른 정답 청크 {unique_positives}개")

    dup_queries = [q for q, count in Counter(queries).items() if count > 1]
    if dup_queries:
        print(f"[경고] 똑같은 질문이 {len(dup_queries)}개 중복되어 있습니다. 데이터 생성 단계를 확인하세요.")

    if unique_positives < len(pairs):
        print(
            "[안내] 같은 정답 청크에 질문이 여러 개 달려 있습니다. "
            "학습 시 같은 청크가 한 묶음(배치)에 같이 들어가지 않도록 NO_DUPLICATES 방식을 사용합니다."
        )

    print("[샘플]")
    for row in pairs[:2]:
        print(f"  질문: {row['query']}")
        print(f"  정답: {row['positive'][:60]}...")


def train(pairs: list[dict], args: argparse.Namespace) -> None:
    """실제 학습을 수행한다. 무거운 라이브러리는 여기서만 불러온다 (--dry-run 에서는 필요 없음)."""
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        print("[경고] GPU를 찾지 못했습니다. CPU로는 매우 느립니다. EC2 서버에서 실행했는지 확인하세요.")
    use_bf16 = has_gpu and torch.cuda.is_bf16_supported()  # A10G 같은 최신 GPU는 bf16 지원
    use_fp16 = has_gpu and not use_bf16

    # 1) 사전학습된 모델 불러오기 (처음 실행하면 인터넷에서 내려받는다)
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length  # 한 번에 읽는 최대 길이 (길수록 메모리를 많이 씀)

    # 2) 학습 데이터를 모델이 읽는 표 형태로 변환 (칸 이름: anchor=질문, positive=정답)
    dataset = Dataset.from_dict(
        {
            "anchor": [p["query"] for p in pairs],
            "positive": [p["positive"] for p in pairs],
        }
    )

    # 3) 손실 함수: 같은 묶음 안의 "다른 질문의 정답 청크"를 오답으로 삼아 비교 학습한다
    loss = MultipleNegativesRankingLoss(model)

    # 4) 학습 설정
    output_dir = Path(args.output_dir)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,  # 전체 데이터를 몇 바퀴 볼지
        per_device_train_batch_size=args.batch_size,  # 한 번에 묶어서 공부하는 개수
        learning_rate=args.learning_rate,  # 한 번에 얼마나 크게 고칠지
        warmup_ratio=0.1,  # 처음 10%는 천천히 시작
        bf16=use_bf16,
        fp16=use_fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # 같은 청크가 한 묶음에 중복되지 않게
        logging_steps=10,  # 10스텝마다 loss(오차)를 로그에 출력
        save_strategy="no",  # 중간 저장은 끄고 마지막에 한 번만 저장 (디스크 절약)
        report_to="none",  # 외부 기록 서비스(wandb 등)에 연결하지 않음
        seed=42,
    )

    # 5) 학습 실행
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )
    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    # 6) 학습된 모델과 학습 요약 저장
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))

    summary = {
        "base_model": args.model_name,
        "num_pairs": len(pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        "elapsed_seconds": round(elapsed, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[완료] 모델 저장: {final_dir}")
    print(f"[완료] 학습 요약: {output_dir / 'training_summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="임베딩 모델 파인튜닝")
    parser.add_argument("--train-file", required=True, help="학습 데이터 .jsonl 경로")
    parser.add_argument("--output-dir", default="ml/output/embedding-v1", help="학습된 모델을 저장할 폴더")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="출발점이 되는 사전학습 모델 이름")
    parser.add_argument("--epochs", type=int, default=3, help="전체 데이터를 반복할 횟수")
    parser.add_argument("--batch-size", type=int, default=16, help="한 번에 묶어서 학습하는 개수")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="학습률")
    parser.add_argument("--max-seq-length", type=int, default=512, help="한 번에 읽는 최대 토큰 길이")
    parser.add_argument("--dry-run", action="store_true", help="데이터 점검만 하고 학습은 하지 않음")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_pairs(Path(args.train_file))
    summarize(pairs)

    if args.dry_run:
        print("[dry-run] 데이터 점검만 마치고 종료합니다. 학습은 하지 않았습니다.")
        return

    if len(pairs) < args.batch_size:
        raise ValueError(
            f"데이터({len(pairs)}건)가 배치 크기({args.batch_size})보다 적습니다. --batch-size 를 줄이세요."
        )
    train(pairs, args)


if __name__ == "__main__":
    main()
