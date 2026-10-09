package com.hsleecoder.ragchatbot.document.repository;

import com.hsleecoder.ragchatbot.document.domain.Document;
import com.hsleecoder.ragchatbot.document.domain.DocumentStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

/** Document 엔티티 조회/저장용 리포지토리. */
public interface DocumentRepository extends JpaRepository<Document, Long> {

    /** 특정 처리 상태의 문서 목록을 조회한다. */
    List<Document> findByStatus(DocumentStatus status);
}