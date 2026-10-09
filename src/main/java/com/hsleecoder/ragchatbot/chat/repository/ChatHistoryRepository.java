package com.hsleecoder.ragchatbot.chat.repository;

import com.hsleecoder.ragchatbot.chat.domain.ChatHistory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

/** ChatHistory 엔티티 조회/저장용 리포지토리. */
public interface ChatHistoryRepository extends JpaRepository<ChatHistory, Long> {

    /** 최신순 페이징 조회. */
    Page<ChatHistory> findAllByOrderByCreatedAtDesc(Pageable pageable);
}