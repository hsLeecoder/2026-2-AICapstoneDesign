"""표준취업규칙 청크가 구간별로 얼마나 겹치는지 확인한다 (읽기만 함, 파일을 바꾸지 않음).

사용 예 (로컬 PowerShell, 프로젝트 루트에서):
    python ml/scripts/check_dups.py
"""
import argparse
import json
import re
from collections import defaultdict


def title_of(heading):
    """'제61조 (퇴직급여제도의 설정)' -> '퇴직급여제도의설정' (조 번호와 공백을 뺀 제목)"""
    m = re.search(r"\(([^)]*)\)", heading)
    return re.sub(r"\s+", "", m.group(1)) if m else re.sub(r"\s+", "", heading)


def main():
    p = argparse.ArgumentParser(description="취업규칙 청크의 구간별 중복 확인")
    p.add_argument("--chunks", default="ml/data/chunks/chunks.jsonl")
    p.add_argument("--doc-prefix", default="workrules")
    p.add_argument("--a-end", type=int, default=40, help="앞 구간의 마지막 페이지")
    p.add_argument("--b-start", type=int, default=47, help="뒤 구간의 시작 페이지")
    a = p.parse_args()

    rows = []
    with open(a.chunks, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["doc"].startswith(a.doc_prefix):
                    rows.append(r)

    front = [r for r in rows if r["page"] <= a.a_end]
    middle = [r for r in rows if a.a_end < r["page"] < a.b_start]
    back = [r for r in rows if r["page"] >= a.b_start]

    def marked(rs):
        n = sum(1 for r in rs if "[필수]" in r["text"] or "[선택]" in r["text"])
        return f"{n}/{len(rs)}" if rs else "0/0"

    print(f"[구간별 청크 수] 앞(p.1~{a.a_end}) {len(front)}개 / 중간(p.{a.a_end + 1}~{a.b_start - 1}) {len(middle)}개 / 뒤(p.{a.b_start}~) {len(back)}개")
    print(f"[해설 표시([필수]·[선택])가 들어간 청크] 앞 {marked(front)} / 중간 {marked(middle)} / 뒤 {marked(back)}")

    print(f"\n[중간 구간 조항 제목] (앞뒤와 다른 문서인지 확인용)")
    for r in middle[:25]:
        print(f"  p.{r['page']} {r['heading']}")
    if len(middle) > 25:
        print(f"  ... 외 {len(middle) - 25}개")

    fp, bp = defaultdict(list), defaultdict(list)
    for r in front:
        fp[title_of(r["heading"])].append(r["page"])
    for r in back:
        bp[title_of(r["heading"])].append(r["page"])
    both = sorted(set(fp) & set(bp), key=lambda t: min(fp[t]))
    print(f"\n[앞·뒤 구간에 모두 나오는 조항 제목] {len(both)}개 (앞 구간 제목 {len(fp)}개 중)")
    for t in both[:8]:
        print(f"  {t}: 앞 p.{sorted(set(fp[t]))} / 뒤 p.{sorted(set(bp[t]))}")


if __name__ == "__main__":
    main()
