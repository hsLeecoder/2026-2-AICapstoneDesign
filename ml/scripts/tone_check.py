"""답변 말투를 센다: 문장 끝이 '~니다/~요'(존댓말)인지 '~다'(평어)인지.
실행 위치: 로컬 PowerShell (레포 루트)
예) python ml/scripts/tone_check.py v2=ml/data/eval/answers_v2.jsonl v3=ml/data/eval/answers_v3.jsonl
거절 답변("확인되지 않습니다")은 제외하고, 한 답변에 존댓말과 평어가 섞였으면 '혼용'으로 센다.
"""
import json, re, sys

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]

def kind(sent):
    s = re.sub(r"[\s.!?\"'”’)\]]+$", "", sent)
    if re.search(r"(니다|세요|어요|아요|해요|예요|이에요|에요|죠|까요|나요|네요)$", s):
        return "polite"
    if re.search(r"다$", s):
        return "plain"
    return None

def main():
    if len(sys.argv) < 2:
        print("사용법: python ml/scripts/tone_check.py 이름=파일 이름=파일 ..."); return
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        n = pol = pla = mix = 0
        mixed_ids = []
        for r in rows:
            a = r.get("answer") or ""
            if not a or r.get("refused") or r.get("answered") is False or "확인되지 않" in a:
                continue
            ks = {kind(s) for s in sentences(a)} - {None}
            if not ks:
                continue
            n += 1
            if ks == {"polite"}: pol += 1
            elif ks == {"plain"}: pla += 1
            else:
                mix += 1; mixed_ids.append(r.get("id"))
        print(f"{name}: 답변 {n}개 중 존댓말만 {pol}, 평어만 {pla}, 혼용 {mix}" + (f"  혼용 질문: {', '.join(map(str, mixed_ids))}" if mixed_ids else ""))

if __name__ == "__main__":
    main()
