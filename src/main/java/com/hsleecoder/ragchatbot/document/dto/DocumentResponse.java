package com.hsleecoder.ragchatbot.document.dto;

import com.hsleecoder.ragchatbot.document.domain.Document;
import com.hsleecoder.ragchatbot.document.domain.DocumentStatus;

import java.time.LocalDateTime;

/**
 * 문서 정보를 프론트엔드로 내려줄 때 쓰는 응답 DTO.
 * 엔티티를 그대로 노출하지 않기 위해 별도 record 로 분리한다.
 */
public record DocumentResponse(
        Long id,
        String originalFileName,
        long fileSize,
        DocumentStatus status,
        LocalDateTime createdAt
) {
    /** Document 엔티티를 응답 DTO 로 변환한다. */
    public static DocumentResponse from(Document document) {
        return new DocumentResponse(
                document.getId(),
                document.getOriginalFileName(),
                document.getFileSize(),
                document.getStatus(),
                document.getCreatedAt()
        );
    }
}