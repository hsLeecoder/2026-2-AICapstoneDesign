"""문제집(train.jsonl)을 학습용 / 검증용으로 나눈다.

청크마다 질문 N개 중 일부(기본 1개)를 검증용으로 빼 둔다.
검증용 질문은 학습에 쓰이지 않으므로, 학습 전·후 점수를 공정하게 비교할 수 있다.

사용 예 (로컬 PowerShell, 프로젝트 루트에서):
    python ml/scripts/split_pairs.py
"""
import argparse
import json
import random
from collections import defaultdict


def main():
    p = argparse.ArgumentParser(description="문제집을 학습용/검증용으로 분리")
    p.add_argument("--input", default="ml/data/pairs/train.jsonl", help="전체 문제집 .jsonl")
    p.add_argument("--train-out", default="ml/data/pairs/train_split.jsonl", help="학습용 저장 경로")
    p.add_argument("--valid-out", default="ml/data/pairs/valid.jsonl", help="검증용 저장 경로")
    p.add_argument("--valid-per-chunk", type=int, default=1, help="청크당 검증용으로 뺄 질문 수")
    p.add_argument("--seed", type=int, default=42, help="무작위 시드 (같으면 같은 결과)")
    a = p.parse_args()

    groups = defaultdict(list)
    with open(a.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                groups[row["chunk_id"]].append(row)

    rng = random.Random(a.seed)
    train, valid = [], []
    for cid in sorted(groups):
        rows = groups[cid][:]
        rng.shuffle(rows)
        # 질문이 너무 적은 청크는 전부 학습용으로 둔다
        n_valid = a.valid_per_chunk if len(rows) > a.valid_per_chunk else 0
        valid.extend(rows[:n_valid])
        train.extend(rows[n_valid:])
    rng.shuffle(train)

    for path, rows in ((a.train_out, train), (a.valid_out, valid)):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[완료] 청크 {len(groups)}개 / 전체 질문 {len(train) + len(valid)}개")
    print(f"  학습용 {len(train)}개 -> {a.train_out}")
    print(f"  검증용 {len(valid)}개 -> {a.valid_out}")


if __name__ == "__main__":
    main()
