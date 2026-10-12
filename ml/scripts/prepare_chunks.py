"""PDF → 청크(조각) 변환 스크립트.

하는 일:
    규정 PDF에서 글자를 뽑아내고, "제N조(제목)" 단위로 잘라서 청크 목록(.jsonl)으로 저장한다.
    너무 긴 조항은 글자 수 기준으로 다시 나눈다. 이 단계에는 OpenAI 키가 필요 없다.
    실행하면 PDF별로 "글자가 제대로 뽑혔는지, 스캔본은 아닌지" 점검 결과도 같이 보여준다.

사용 예:
    # 1) 점검만 (파일을 만들지 않고 결과만 확인)
    python ml/scripts/prepare_chunks.py --pdf-dir ml/data/raw_pdf --report-only

    # 2) 청크 파일 만들기
    python ml/scripts/prepare_chunks.py --pdf-dir ml/data/raw_pdf --output ml/data/chunks/train_chunks.jsonl

청크 한 줄의 모양:
    {"chunk_id": "welfare_ananti-003", "doc": "welfare_ananti.pdf", "page": 5,
     "heading": "제12조(경조사비)", "text": "제12조(경조사비) 본인 결혼 시 ..."}

주의: 청크 파일에는 규정 원문이 그대로 들어 있으므로 GitHub에 올리지 않는다 (.gitignore 대상).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_PDF_DIR = "ml/data/raw_pdf"
DEFAULT_OUTPUT = "ml/data/chunks/train_chunks.jsonl"

# "제12조(경조사비)" 처럼 조 번호 뒤에 괄호 제목이 붙은 줄을 조항의 시작으로 본다.
# (본문 안에서 "제5조에 따라..." 처럼 다른 조항을 인용한 줄과 구분하기 위해 괄호 제목을 필수로 한다)
ARTICLE_RE = re.compile(r"^\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*[\(（]([^)）]{1,40})[\)）]")
# "제1장 총칙", "제2절 휴가" 같은 장/절 제목 줄 (짧은 줄만)
CHAPTER_RE = re.compile(r"^\s*제\s*\d+\s*(?:장|절)")
# 줄 전체가 숫자 하나뿐인 경우는 쪽 번호로 보고 버린다
PAGE_NUMBER_RE = re.compile(r"^\s*[-–]?\s*\d{1,3}\s*[-–]?\s*$")
# 줄 끝이 이 글자면 문장이 끝난 것으로 보고 다음 줄은 줄바꿈으로 이어 붙인다
SENTENCE_END = (".", ":", "?", "!", "。")


def extract_lines(pdf_path: Path) -> tuple[list[tuple[int, str]], dict]:
    """PDF에서 (쪽 번호, 한 줄) 목록과 점검용 통계를 뽑는다."""
    import pdfplumber  # 이 단계에서만 필요하므로 여기서 불러온다

    lines: list[tuple[int, str]] = []
    empty_pages = 0
    total_chars = 0
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < 30:  # 글자가 거의 없는 쪽 (그림만 있거나 스캔본일 수 있음)
                empty_pages += 1
            for raw in text.splitlines():
                line = raw.strip()
                if not line or PAGE_NUMBER_RE.match(line):
                    continue
                lines.append((page_no, line))
                total_chars += len(line)
    return lines, {"pages": num_pages, "empty_pages": empty_pages, "chars": total_chars}


def join_lines(lines: list[str]) -> str:
    """PDF는 문장 중간에서 줄이 바뀌므로, 문장이 끝난 곳만 줄바꿈으로 두고 나머지는 공백으로 잇는다."""
    result = ""
    for line in lines:
        if not result:
            result = line
        elif result.endswith(SENTENCE_END):
            result += "\n" + line
        else:
            result += " " + line
    return result


def split_into_articles(lines: list[tuple[int, str]]) -> list[dict]:
    """줄 목록을 조항(제N조) 단위 덩어리로 나눈다. 조항 표시가 없으면 통째로 한 덩어리가 된다."""
    blocks: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current and current["lines"]:
            blocks.append(current)
        current = None

    for page_no, line in lines:
        match = ARTICLE_RE.match(line)
        if match:
            flush()
            number = f"제{match.group(1)}조" + (f"의{match.group(2)}" if match.group(2) else "")
            current = {"heading": f"{number}({match.group(3).strip()})", "lines": [(page_no, line)]}
            continue
        if CHAPTER_RE.match(line) and len(line) <= 30:
            flush()  # 장/절 제목은 조각에 넣지 않고 경계로만 사용
            continue
        if current is None:
            current = {"heading": "", "lines": []}
        current["lines"].append((page_no, line))
    flush()
    return blocks


def pack_block(block: dict, max_chars: int) -> list[dict]:
    """한 덩어리가 max_chars보다 길면 여러 조각으로 나눈다. 둘째 조각부터는 제목을 앞에 붙여 문맥을 유지한다."""
    pieces: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_len = 0
    for page_no, line in block["lines"]:
        # 한 줄 자체가 너무 길면 강제로 자른다
        parts = [line[i : i + max_chars] for i in range(0, len(line), max_chars)] if len(line) > max_chars else [line]
        for part in parts:
            if current and current_len + len(part) + 1 > max_chars:
                pieces.append(current)
                current, current_len = [], 0
            current.append((page_no, part))
            current_len += len(part) + 1
    if current:
        pieces.append(current)

    chunks = []
    for index, piece in enumerate(pieces):
        text = join_lines([line for _, line in piece])
        if index > 0 and block["heading"]:
            text = f"{block['heading']} (계속)\n{text}"
        chunks.append({"heading": block["heading"], "page": piece[0][0], "text": text})
    return chunks


def process_pdf(pdf_path: Path, max_chars: int, min_chars: int) -> tuple[list[dict], dict]:
    """PDF 하나를 청크 목록으로 바꾼다."""
    lines, stats = extract_lines(pdf_path)
    blocks = split_into_articles(lines)

    chunks: list[dict] = []
    dropped = 0
    for block in blocks:
        for chunk in pack_block(block, max_chars):
            if len(chunk["text"]) < min_chars:  # 너무 짧은 조각(목차 줄 등)은 학습에 도움이 안 되어 제외
                dropped += 1
                continue
            chunks.append(chunk)

    for index, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = f"{pdf_path.stem}-{index:03d}"
        chunk["doc"] = pdf_path.name

    stats.update(
        {
            "articles": sum(1 for b in blocks if b["heading"]),
            "chunks": len(chunks),
            "dropped": dropped,
        }
    )
    return chunks, stats


def print_report(pdf_name: str, stats: dict, chunks: list[dict]) -> None:
    """사람이 눈으로 확인할 수 있게 PDF별 점검 결과를 보여준다."""
    print(f"\n=== {pdf_name} ===")
    print(f"  쪽 수 {stats['pages']} (글자가 거의 없는 쪽 {stats['empty_pages']}) / 뽑힌 글자 수 {stats['chars']}")
    print(f"  조항(제N조) 감지 {stats['articles']}개 / 청크 {stats['chunks']}개 (짧아서 제외 {stats['dropped']}개)")
    if chunks:
        lengths = [len(c["text"]) for c in chunks]
        print(f"  청크 길이: 평균 {sum(lengths) // len(lengths)}자, 최대 {max(lengths)}자")

    # 경고: 사람이 확인해야 할 상황
    if stats["pages"] and stats["chars"] / stats["pages"] < 100:
        print("  [경고] 쪽당 글자 수가 매우 적습니다. 스캔본(이미지) PDF일 수 있어 이 문서는 제외를 권장합니다.")
    if stats["empty_pages"] > stats["pages"] / 2:
        print("  [경고] 글자가 거의 없는 쪽이 절반 이상입니다.")
    if stats["articles"] == 0 and chunks:
        print("  [안내] '제N조(제목)' 형식이 감지되지 않아 글자 수 기준으로만 나눴습니다. 샘플을 꼭 확인하세요.")

    for chunk in chunks[:2]:
        preview = chunk["text"].replace("\n", " ")[:100]
        print(f"  [샘플 p.{chunk['page']}] {preview}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF를 청크(조각) 파일로 변환")
    parser.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR, help="PDF가 들어 있는 폴더")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="청크를 저장할 .jsonl 경로")
    parser.add_argument("--max-chars", type=int, default=700, help="청크 하나의 최대 글자 수")
    parser.add_argument("--min-chars", type=int, default=40, help="이보다 짧은 조각은 제외")
    parser.add_argument("--report-only", action="store_true", help="점검 결과만 보고 파일은 만들지 않음")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_dir}")

    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        chunks, stats = process_pdf(pdf_path, args.max_chars, args.min_chars)
        print_report(pdf_path.name, stats, chunks)
        all_chunks.extend(chunks)

    print(f"\n[합계] PDF {len(pdf_files)}개 → 청크 {len(all_chunks)}개")
    if args.report_only:
        print("[report-only] 점검만 하고 파일은 만들지 않았습니다.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[완료] 청크 파일 저장: {output}")


if __name__ == "__main__":
    main()
