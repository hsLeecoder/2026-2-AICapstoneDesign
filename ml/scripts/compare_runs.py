"""채점된 여러 실행 결과(graded jsonl)를 한 표로 비교한다.
실행 위치: 로컬 PowerShell (레포 루트)
예) python ml/scripts/compare_runs.py v1_k3=ml/data/eval/graded_v1_k3.jsonl v2_k3=ml/data/eval/graded_v2_k3.jsonl
첫 번째 실행을 기준(개선 전)으로 삼아 나머지가 좋아졌는지/나빠졌는지 센다.
"""
import json, sys

def load(path):
    return {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}

def main():
    if len(sys.argv) < 3:
        print("사용법: python ml/scripts/compare_runs.py 이름=파일 이름=파일 ..."); return
    runs = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        runs.append((name, load(path)))
    base_name, base = runs[0]
    ids = list(base.keys())

    def stats(rows, types):
        sc = [r["rag_score"] for r in rows if r["type"] in types and r.get("rag_score") is not None]
        if not sc: return "-"
        return f"{sum(sc)/len(sc):.2f} (2점 {sum(1 for s in sc if s==2)}/{len(sc)})"

    groups = [("규정 질문(생성)", lambda r: r["type"] == "in_scope" and r.get("source") == "generated"),
              ("규정 질문(직접 작성)", lambda r: r["type"] == "in_scope" and r.get("source") == "user"),
              ("함정·세부 질문", lambda r: r["type"] == "trap"),
              ("규정 외 질문", lambda r: r["type"] == "out_of_scope")]
    print("=== 그룹별 평균 점수 (2점 만점) ===")
    print("그룹".ljust(18) + "".join(n.ljust(24) for n, _ in runs))
    for g, f in groups:
        line = g.ljust(16)
        for n, rows in runs:
            sc = [r["rag_score"] for r in rows.values() if f(r) and r.get("rag_score") is not None]
            line += (f"{sum(sc)/len(sc):.2f} (2점 {sum(1 for s in sc if s==2)}/{len(sc)})" if sc else "-").ljust(24)
        print(line)

    print("\n=== 기준(" + base_name + ") 대비 변화 ===")
    for n, rows in runs[1:]:
        up = [i for i in ids if i in rows and (rows[i].get("rag_score") or 0) > (base[i].get("rag_score") or 0)]
        down = [i for i in ids if i in rows and (rows[i].get("rag_score") or 0) < (base[i].get("rag_score") or 0)]
        print(f"[{n}] 좋아진 질문 {len(up)}개, 나빠진 질문 {len(down)}개")
        for tag, lst in (("  ▲", up), ("  ▼", down)):
            for i in lst:
                print(f"{tag} {i} {base[i]['question'][:30]} : {base[i].get('rag_score')} → {rows[i].get('rag_score')}  ({rows[i].get('rag_reason','')[:60]})")

    print("\n=== 실행마다 점수가 다른 질문 ===")
    for i in ids:
        sc = [rows[i].get("rag_score") if i in rows else None for _, rows in runs]
        if len(set(sc)) > 1:
            print(f"{i} {base[i]['question'][:28].ljust(30)} " + " / ".join(f"{n}={s}" for (n, _), s in zip(runs, sc)))

if __name__ == "__main__":
    main()
