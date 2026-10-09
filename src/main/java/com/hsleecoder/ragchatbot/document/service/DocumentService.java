package com.hsleecoder.ragchatbot.document.service;

import com.hsleecoder.ragchatbot.document.domain.Document;
import com.hsleecoder.ragchatbot.document.dto.DocumentResponse;
import com.hsleecoder.ragchatbot.document.repository.DocumentRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

/**
 * PDF 문서 업로드 및 조회를 담당하는 서비스.
 * TODO(다음 턴): PDFBox 텍스트 추출 → 청킹 → 임베딩 → Qdrant 적재 로직을 여기에 이어 붙인다.
 */
@Service
@RequiredArgsConstructor
public class DocumentService {

    private final DocumentRepository documentRepository;

    @Value("${app.upload-dir}")
    private String uploadDir;

    /** PDF 파일을 로컬에 저장하고 메타데이터를 DB에 기록한다. */
    public DocumentResponse upload(MultipartFile file) {
        validate(file);

        String storedFileName = UUID.randomUUID() + ".pdf";
        saveToDisk(file, storedFileName);

        Document document = Document.builder()
                .originalFileName(file.getOriginalFilename())
                .storedFileName(storedFileName)
                .contentType(file.getContentType())
                .fileSize(file.getSize())
                .build();

        return DocumentResponse.from(documentRepository.save(document));
    }

    /** 등록된 문서 전체 목록을 최신순으로 조회한다. */
    public List<DocumentResponse> findAll() {
        return documentRepository.findAll().stream()
                .sorted((a, b) -> b.getCreatedAt().compareTo(a.getCreatedAt()))
                .map(DocumentResponse::from)
                .toList();
    }

    /** 업로드 파일이 비어있지 않은 PDF 인지 검증한다. */
    private void validate(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("업로드할 파일이 비어 있습니다.");
        }
        String fileName = file.getOriginalFilename();
        boolean isPdf = fileName != null && fileName.toLowerCase().endsWith(".pdf");
        if (!isPdf) {
            throw new IllegalArgumentException("PDF 파일만 업로드할 수 있습니다.");
        }
    }

    /** 업로드 폴더가 없으면 생성한 뒤 파일을 저장한다. */
    private void saveToDisk(MultipartFile file, String storedFileName) {
        try {
            Path dir = Path.of(uploadDir);
            Files.createDirectories(dir);
            file.transferTo(dir.resolve(storedFileName));
        } catch (IOException e) {
            throw new UncheckedIOException("파일 저장에 실패했습니다.", e);
        }
    }
}