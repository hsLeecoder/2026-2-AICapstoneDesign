"""청크 파일에서 중복 구간(특정 문서의 특정 페이지 범위)을 빼고 새 파일로 저장한다.
원본 청크 파일은 바꾸지 않는다.

사용 예 (로컬 PowerShell, 프로젝트 루트에서):
    python ml/scripts/filter_chunks.py --input ml/data/chunks/chunks.jsonl \
        --output ml/data/chunks/chunks_dedup.jsonl --doc-prefix workrules --drop-pages 46-82
"""
import argparse
import json
from collections import Counter


def parse_ranges(text):
    ranges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        ranges.append((int(lo), int(hi) if hi else int(lo)))
    return ranges


def main():
    p = argparse.ArgumentParser(description="청크 파일에서 중복 페이지 구간 제외")
    p.add_argument("--input", default="ml/data/chunks/chunks.jsonl")
    p.add_argument("--output", default="ml/data/chunks/chunks_dedup.jsonl")
    p.add_argument("--doc-prefix", required=True, help="제외를 적용할 문서 이름의 앞부분 (예: workrules)")
    p.add_argument("--drop-pages", required=True, help="제외할 페이지 범위 (예: 46-82 또는 46-82,90)")
    a = p.parse_args()

    ranges = parse_ranges(a.drop_pages)
    kept, dropped = [], []
    with open(a.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            hit = r["doc"].startswith(a.doc_prefix) and any(lo <= r["page"] <= hi for lo, hi in ranges)
            (dropped if hit else kept).append(r)

    with open(a.output, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[완료] 전체 {len(kept) + len(dropped)}개 중 {len(dropped)}개 제외, {len(kept)}개 저장 -> {a.output}")
    for doc, n in sorted(Counter(r["doc"] for r in kept).items()):
        print(f"  {doc}: {n}개")


if __name__ == "__main__":
    main()
