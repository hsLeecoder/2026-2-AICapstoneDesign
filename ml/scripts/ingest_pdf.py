"""PDF 한 개를 청크로 바꿔 Qdrant 컬렉션에 추가·교체·삭제한다 (문서 추가용. 관리자 업로드 기능의 핵심 로직).

모델 학습은 필요 없다. 기존 임베딩 모델로 청크를 벡터로 바꿔 넣기만 한다.
기존 prepare_chunks.py 의 조항 분리 코드를 그대로 쓰고, 글자가 깨지는 PDF(pdfplumber 가 (cid:123) 으로 읽는 경우)는
PyMuPDF 로 다시 읽으며 글자 위치로 띄어쓰기를 복원한다.

사용 예
  # 1) 로컬에서 파싱 결과만 확인 (GPU·Qdrant·모델 필요 없음, 파일을 만들지 않음)
  python ml/scripts/ingest_pdf.py ml/data/raw_pdf/welfare_goyongnodong.pdf --dry-run --drop-pages 7-8 --preview 20

  # 2) EC2 에서 Qdrant 에 추가 (같은 문서가 이미 있으면 그 문서의 청크만 지우고 새로 넣는다)
  python ml/scripts/ingest_pdf.py ml/data/raw_pdf/welfare_goyongnodong.pdf --drop-pages 7-8 --query "선택적 복지가 뭐예요?"

  # 3) 문서 하나만 삭제
  python ml/scripts/ingest_pdf.py --delete-only --doc-name welfare_goyongnodong
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_chunks as pc  # noqa: E402
from filter_chunks import parse_ranges  # noqa: E402

CID_RE = re.compile(r"\(cid:\d+\)")


def garbage_ratio(text: str) -> float:
    """글자가 (cid:123) 처럼 깨진 비율. pdfplumber 가 글꼴 정보를 못 읽을 때 생긴다."""
    return sum(len(m) for m in CID_RE.findall(text)) / max(len(text), 1)


def extract_lines_pymupdf(pdf_path: Path, space_gap: float = 0.2):
    """PyMuPDF 로 (쪽, 줄) 목록과 쪽별 제목을 뽑는다. 글자 사이 간격이 글자 크기의 space_gap 배보다 넓으면 띄어쓰기를 넣는다."""
    try:
        import pymupdf
    except ImportError:
        raise SystemExit("이 PDF는 PyMuPDF 로 읽어야 합니다. 먼저 설치하세요: pip install pymupdf")
    lines, titles, stats = [], {}, {"pages": 0, "chars": 0}
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            stats["pages"] += 1
            rows = []
            for block in page.get_text("rawdict")["blocks"]:
                for ln in block.get("lines", []):
                    text, prev, size = "", None, 0.0
                    for span in ln["spans"]:
                        size = max(size, span["size"])
                        for ch in span["chars"]:
                            x0, _, x1, _ = ch["bbox"]
                            if prev is not None and (x0 - prev) > space_gap * span["size"] and ch["c"] != " ":
                                text += " "
                            text += ch["c"]
                            prev = x1
                    text = text.strip()
                    if text:
                        rows.append((ln["bbox"][1], ln["bbox"][0], size, text))
            rows.sort(key=lambda r: (round(r[0] / 4), r[1]))
            body = [r for r in rows if not pc.PAGE_NUMBER_RE.match(r[3]) and not pc.LEADER_RE.search(r[3])]
            if body:  # 쪽 제목 = 가장 큰 글자의 짧은 줄 (본문 평균보다 1.25배 이상 클 때만)
                sizes = sorted(r[2] for r in body)
                median = sizes[len(sizes) // 2]
                top = max(body, key=lambda r: r[2])
                if top[2] >= 1.25 * median and len(top[3]) <= 30:
                    tokens = top[3].split()  # 'E A P' 처럼 글자 사이가 벌어진 제목은 붙인다
                    titles[page_no] = "".join(tokens) if len(tokens) >= 2 and all(len(t) == 1 for t in tokens) else top[3]
                    body = [r for r in body if r is not top]
            for _, _, _, text in body:
                lines.append((page_no, text))
                stats["chars"] += len(text)
    stats.update({"empty_pages": 0, "toc_lines": 0, "split_sheets": 0})
    return lines, titles, stats


def build_chunks(lines, titles, doc_stem, max_chars, min_chars, drop_ranges, drop_line_res):
    """줄 목록을 청크로 만든다. '제N조(제목)'이 있으면 조항 단위, 없으면 쪽 단위로 나눈 뒤 글자 수로 자른다."""
    lines = [(p, t) for p, t in lines
             if not any(lo <= p <= hi for lo, hi in drop_ranges) and not any(r.search(t) for r in drop_line_res)]
    blocks = pc.split_into_articles(lines)
    has_articles = any(b["heading"] for b in blocks)
    if not has_articles:  # 조항 구조가 없으면 쪽마다 따로 묶어 서로 다른 주제가 한 청크에 섞이지 않게 한다
        by_page = {}
        for p, t in lines:
            by_page.setdefault(p, []).append((p, t))
        blocks = [{"heading": "", "lines": v} for v in by_page.values()]
    chunks = []
    for block in blocks:
        for ch in pc.pack_block(block, max_chars):
            if has_articles and not block["heading"]:
                continue  # 조항 앞의 표지·안내문은 규정 내용이 아니다
            heading = ch["heading"] or titles.get(ch["page"], "")
            text = ch["text"]
            if not ch["heading"] and heading and not text.startswith(heading):
                text = f"{heading}\n{text}"  # 조항 제목이 없는 문서는 쪽 제목을 청크 앞에 붙여 무슨 주제인지 알 수 있게 한다
            if len(text) >= min_chars:
                chunks.append({"heading": heading, "page": ch["page"], "text": text})
    for i, ch in enumerate(chunks, start=1):
        ch["chunk_id"] = f"{doc_stem}-{i:03d}"
        ch["doc"] = f"{doc_stem}.pdf"
    return chunks, has_articles


def all_points(client, collection):
    pts, offset = [], None
    while True:
        batch, offset = client.scroll(collection, limit=500, offset=offset, with_payload=True, with_vectors=False)
        pts += batch
        if offset is None:
            return pts


def ingest(client, collection, chunks, vectors, dim, replace=True, create_if_missing=False):
    """청크를 컬렉션에 넣는다. replace 이면 같은 문서(doc)의 기존 청크를 먼저 지운다. 다른 문서는 건드리지 않는다."""
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
    if not client.collection_exists(collection):
        if not create_if_missing:
            raise SystemExit(f"[중단] 컬렉션 '{collection}'이 없습니다. 새로 만들려면 --create-if-missing 을 붙이세요.")
        client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    else:
        size = client.get_collection(collection).config.params.vectors.size
        if size != dim:
            raise SystemExit(f"[중단] 컬렉션 벡터 차원({size})과 모델 차원({dim})이 다릅니다. 같은 모델로 만든 컬렉션이 맞는지 확인하세요.")
    doc = chunks[0]["doc"]
    if replace:
        client.delete(collection, points_selector=Filter(must=[FieldCondition(key="doc", match=MatchValue(value=doc))]))
    ids = [p.id for p in all_points(client, collection) if isinstance(p.id, int)]
    start = (max(ids) + 1) if ids else 0  # 기존 id 와 겹치지 않게 이어서 번호를 붙인다
    points = [PointStruct(id=start + i, vector=vectors[i], payload={k: c[k] for k in ("chunk_id", "doc", "page", "heading", "text")})
              for i, c in enumerate(chunks)]
    for s in range(0, len(points), 100):
        client.upsert(collection_name=collection, points=points[s:s + 100])
    return len(points)


def delete_doc(client, collection, doc):
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    before = Counter(p.payload["doc"] for p in all_points(client, collection)).get(doc, 0)
    client.delete(collection, points_selector=Filter(must=[FieldCondition(key="doc", match=MatchValue(value=doc))]))
    return before


def doc_counts(client, collection):
    return Counter(p.payload["doc"] for p in all_points(client, collection))


def main():
    p = argparse.ArgumentParser(description="PDF 한 개를 Qdrant 에 추가·교체·삭제")
    p.add_argument("pdf", nargs="?", help="PDF 파일 경로 (--delete-only 일 때는 생략)")
    p.add_argument("--doc-name", help="문서 이름(확장자 제외, 영문 권장). 청크 id 와 출처 파일명에 쓰인다. 기본은 PDF 파일 이름")
    p.add_argument("--dry-run", action="store_true", help="파싱 결과만 보고 모델·Qdrant 는 쓰지 않음")
    p.add_argument("--extractor", choices=["auto", "pdfplumber", "pymupdf"], default="auto")
    p.add_argument("--space-gap", type=float, default=0.2, help="PyMuPDF 읽기에서 띄어쓰기를 넣는 글자 간격 기준(글자 크기 배수)")
    p.add_argument("--column-split", action="store_true", help="표준취업규칙처럼 2단(조문|해설) PDF 일 때만 켬")
    p.add_argument("--max-chars", type=int, default=700)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--drop-pages", default="", help="제외할 PDF 쪽 범위 (예: 7-8 또는 1,7-8)")
    p.add_argument("--drop-line-regex", action="append", default=[], help="이 정규식에 맞는 줄은 제외 (여러 번 사용 가능)")
    p.add_argument("--preview", type=int, default=3, help="청크 미리보기 개수")
    p.add_argument("--save-chunks", help="청크를 이 .jsonl 로 저장 (GitHub 에 올리지 않는다)")
    p.add_argument("--force", action="store_true", help="글자가 깨진 청크가 있어도 진행")
    p.add_argument("--model", default="ml/output/bge-m3-finetuned-v3/final")
    p.add_argument("--collection", default="welfare_chunks_bgem3")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--append", action="store_true", help="같은 문서의 기존 청크를 지우지 않고 추가만 함")
    p.add_argument("--create-if-missing", action="store_true")
    p.add_argument("--delete-only", action="store_true", help="--doc-name 문서의 청크만 삭제")
    p.add_argument("--query", help="적재 후 검색 확인용 질문")
    a = p.parse_args()

    if a.delete_only:
        if not a.doc_name:
            raise SystemExit("--delete-only 에는 --doc-name 이 필요합니다.")
        from qdrant_client import QdrantClient
        n = delete_doc(QdrantClient(url=a.qdrant_url), a.collection, f"{a.doc_name}.pdf")
        print(f"[삭제 완료] {a.doc_name}.pdf 청크 {n}개를 '{a.collection}'에서 지웠습니다.")
        return
    if not a.pdf:
        raise SystemExit("PDF 경로를 알려주세요.")
    pdf_path = Path(a.pdf)
    stem = a.doc_name or pdf_path.stem
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", stem):
        raise SystemExit(f"문서 이름 '{stem}'에 영문·숫자·_·- 외의 글자가 있습니다. --doc-name 으로 영문 이름을 정해주세요.")

    # 1) 글자 읽기: pdfplumber 로 읽어 깨졌으면 PyMuPDF 로 다시 읽는다
    titles, extractor = {}, a.extractor
    if extractor in ("auto", "pdfplumber"):
        lines, stats = pc.extract_lines(pdf_path, a.column_split)
        ratio = garbage_ratio("\n".join(t for _, t in lines))
        if extractor == "auto" and ratio > 0.02:
            print(f"[안내] pdfplumber 로 읽은 글자의 {ratio:.0%}가 (cid:..) 로 깨졌습니다. PyMuPDF 로 다시 읽습니다.")
            extractor = "pymupdf"
        else:
            extractor = "pdfplumber"
    if extractor == "pymupdf":
        lines, titles, stats = extract_lines_pymupdf(pdf_path, a.space_gap)
    chunks, has_articles = build_chunks(lines, titles, stem, a.max_chars, a.min_chars, parse_ranges(a.drop_pages),
                                        [re.compile(r) for r in a.drop_line_regex])
    if not chunks:
        raise SystemExit("[중단] 청크가 하나도 만들어지지 않았습니다. 스캔본(이미지) PDF일 수 있습니다.")

    # 2) 점검 결과
    bad = [c for c in chunks if garbage_ratio(c["text"]) > 0.01]
    lengths = [len(c["text"]) for c in chunks]
    print(f"\n=== {pdf_path.name} → 문서 이름 {stem}.pdf ===")
    print(f"  읽기 방식 {extractor} / 쪽 수 {stats['pages']} / 조항(제N조) 구조 {'있음' if has_articles else '없음 → 쪽 단위로 나눔'}")
    print(f"  청크 {len(chunks)}개 (길이 평균 {sum(lengths) // len(lengths)}자, 최대 {max(lengths)}자)")
    if titles:
        print("  쪽 제목: " + ", ".join(f"p.{k} {v}" for k, v in sorted(titles.items())))
    for c in chunks[:a.preview]:
        print(f"  [{c['chunk_id']} p.{c['page']} | {c['heading'] or '제목 없음'}] " + c["text"].replace("\n", " / ")[:150])
    if bad:
        print(f"  [경고] 글자가 깨진 청크가 {len(bad)}개 있습니다.")
        if not a.force:
            raise SystemExit("[중단] 깨진 글자를 넣으면 검색이 망가집니다. 원인을 확인하거나 --force 로 강제하세요.")
    if a.save_chunks:
        Path(a.save_chunks).parent.mkdir(parents=True, exist_ok=True)
        with open(a.save_chunks, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  청크 저장: {a.save_chunks}")
    if a.dry_run:
        print("[dry-run] 파싱 결과만 확인했고 Qdrant 에는 넣지 않았습니다.")
        return

    # 3) 임베딩 + 적재
    import torch
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(a.model, device=device)
    model.max_seq_length = 512
    vectors = model.encode([c["text"] for c in chunks], batch_size=a.batch_size, normalize_embeddings=True, show_progress_bar=False)
    client = QdrantClient(url=a.qdrant_url)
    n = ingest(client, a.collection, chunks, [v.tolist() for v in vectors], model.get_sentence_embedding_dimension(),
               replace=not a.append, create_if_missing=a.create_if_missing)
    print(f"[적재 완료] {stem}.pdf 청크 {n}개 → '{a.collection}'. 문서별 청크 수: {dict(doc_counts(client, a.collection))}")
    if a.query:
        qv = model.encode([a.query], normalize_embeddings=True)[0].tolist()
        print(f"\n[검색 확인] 질문: {a.query}")
        for r in client.query_points(a.collection, query=qv, limit=3, with_payload=True).points:
            pl = r.payload
            print(f"  점수 {r.score:.3f} | {pl['doc']} p.{pl['page']} | {pl['heading']} | {pl['text'][:60]}...")


if __name__ == "__main__":
    main()
