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
PAGE_NUMBER_RE = re.compile(r"^\s*[-–−—]?\s*\d{1,3}\s*[-–−—]?\s*$")
# 목차 줄은 "제1조(목적) ········ 7" 처럼 점선(······)으로 쪽 번호가 이어진다. 내용이 아니므로 버린다
LEADER_RE = re.compile(r"[·∙⋅・…⋯]{3,}|\.{5,}")
# 줄 끝이 이 글자면 문장이 끝난 것으로 보고 다음 줄은 줄바꿈으로 이어 붙인다
SENTENCE_END = (".", ":", "?", "!", "。")

# 표준취업규칙 PDF는 한 쪽에 시트가 위아래로 2장 인쇄되어 있고, 각 시트는 "취업규칙(안)(조문) | (작성시 착안사항)(해설)" 2단이다.
# 두 단을 그냥 읽으면 해설이 조문 사이에 섞여 청크가 오염되므로, 해설 단은 버리고 조문 단만 읽는다.
LEFT_HEADER = "취업규칙(안)"
RIGHT_HEADER = "(작성시"
DEFAULT_SPLIT_X = 350.0  # 조문 단과 해설 단의 경계 x 좌표 (pt)
SHEET_HEADER_OFFSET = 55.0  # 아래 시트의 머리글에서 이만큼 위가 위 시트의 끝


def sheet_regions(page, words: list[dict]) -> list[tuple[float, float, bool]]:
    """쪽을 시트(머리글 기준) 영역으로 나누고, 영역마다 2단(해설 단이 있음)인지 함께 돌려준다."""
    tops = sorted(w["top"] for w in words if w["text"] == LEFT_HEADER)
    clusters: list[float] = []
    for top in tops:  # 같은 높이의 머리글(좌우 단)은 하나로 묶는다
        if not clusters or top - clusters[-1] > 15:
            clusters.append(top)
    cut = None
    if len(clusters) >= 2:
        cut = clusters[1] - SHEET_HEADER_OFFSET
    elif clusters and clusters[0] > 300:
        cut = clusters[0] - SHEET_HEADER_OFFSET
    bounds = [(0.0, cut), (cut, float(page.height))] if cut else [(0.0, float(page.height))]
    return [
        (top, bottom, any(w["text"] == RIGHT_HEADER and top <= w["top"] < bottom for w in words))
        for top, bottom in bounds
    ]


def page_text(page, split_columns: bool, split_x: float) -> tuple[str, int]:
    """쪽의 글자를 뽑는다. 2단 시트는 조문 단(왼쪽)만 읽고, 나머지는 쪽 전체를 읽는다."""
    if not split_columns:
        return page.extract_text() or "", 0
    words = page.extract_words()
    parts: list[str] = []
    two_column_sheets = 0
    for top, bottom, two_column in sheet_regions(page, words):
        right = split_x if two_column else float(page.width)
        parts.append(page.crop((0.0, top, right, bottom)).extract_text() or "")
        two_column_sheets += int(two_column)
    return "\n".join(parts), two_column_sheets


def extract_lines(pdf_path: Path, split_columns: bool = True, split_x: float = DEFAULT_SPLIT_X) -> tuple[list[tuple[int, str]], dict]:
    """PDF에서 (쪽 번호, 한 줄) 목록과 점검용 통계를 뽑는다."""
    import pdfplumber  # 이 단계에서만 필요하므로 여기서 불러온다

    lines: list[tuple[int, str]] = []
    empty_pages = 0
    total_chars = 0
    toc_lines = 0
    split_sheets = 0
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            text, sheets = page_text(page, split_columns, split_x)
            split_sheets += sheets
            if len(text.strip()) < 30:  # 글자가 거의 없는 쪽 (그림만 있거나 스캔본일 수 있음)
                empty_pages += 1
            for raw in text.splitlines():
                line = raw.strip()
                if not line or PAGE_NUMBER_RE.match(line) or line == LEFT_HEADER:
                    continue
                if LEADER_RE.search(line):  # 목차 줄
                    toc_lines += 1
                    continue
                lines.append((page_no, line))
                total_chars += len(line)
    return lines, {
        "pages": num_pages, "empty_pages": empty_pages, "chars": total_chars,
        "toc_lines": toc_lines, "split_sheets": split_sheets,
    }


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


def process_pdf(
    pdf_path: Path, max_chars: int, min_chars: int, keep_preamble: bool = False,
    split_columns: bool = True, split_x: float = DEFAULT_SPLIT_X,
) -> tuple[list[dict], dict]:
    """PDF 하나를 청크 목록으로 바꾼다."""
    lines, stats = extract_lines(pdf_path, split_columns, split_x)
    blocks = split_into_articles(lines)
    # "제N조(제목)" 구조가 있는 문서에서, 조항에 속하지 않는 앞부분(표지, 목차, 작성 안내문)은 규정 내용이 아니다
    has_articles = any(block["heading"] for block in blocks)

    chunks: list[dict] = []
    dropped = 0
    preamble = 0
    for block in blocks:
        for chunk in pack_block(block, max_chars):
            if has_articles and not block["heading"] and not keep_preamble:
                preamble += 1
                continue
            if len(chunk["text"]) < min_chars:  # 너무 짧은 조각은 학습에 도움이 안 되어 제외
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
            "preamble": preamble,
        }
    )
    return chunks, stats


def print_report(pdf_name: str, stats: dict, chunks: list[dict]) -> None:
    """사람이 눈으로 확인할 수 있게 PDF별 점검 결과를 보여준다."""
    print(f"\n=== {pdf_name} ===")
    print(f"  쪽 수 {stats['pages']} (글자가 거의 없는 쪽 {stats['empty_pages']}) / 뽑힌 글자 수 {stats['chars']}")
    if stats.get("preamble"):
        print(f"  조항에 속하지 않는 앞부분(표지, 안내문 등) 조각 {stats['preamble']}개는 규정 내용이 아니라서 제외했습니다.")
    if stats.get("split_sheets"):
        print(f"  2단(조문 | 해설) 시트 {stats['split_sheets']}개에서 해설 단을 제외하고 조문 단만 읽었습니다.")
    if stats.get("toc_lines"):
        print(f"  목차 줄 {stats['toc_lines']}개는 내용이 아니라서 제외했습니다.")
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
    parser.add_argument("--keep-preamble", action="store_true", help="조항 앞의 표지, 안내문 조각도 제외하지 않고 포함")
    parser.add_argument("--no-column-split", action="store_true", help="2단 시트의 해설 단을 제외하지 않고 쪽 전체를 읽음 (예전 방식)")
    parser.add_argument("--column-split-x", type=float, default=DEFAULT_SPLIT_X, help="조문 단과 해설 단의 경계 x 좌표(pt)")
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
        chunks, stats = process_pdf(
            pdf_path, args.max_chars, args.min_chars, args.keep_preamble,
            split_columns=not args.no_column_split, split_x=args.column_split_x,
        )
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
