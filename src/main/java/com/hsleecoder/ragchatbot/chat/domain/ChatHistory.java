package com.hsleecoder.ragchatbot.chat.domain;

import com.hsleecoder.ragchatbot.document.domain.Document;
import com.hsleecoder.ragchatbot.global.entity.BaseTimeEntity;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 사용자 질문과 AI 응답 이력 엔티티.
 * 답변의 근거가 된 문서(참조 문서)를 함께 저장해 출처 추적이 가능하다.
 * (생성 시간은 BaseTimeEntity.createdAt 이 담당)
 */
@Getter
@Entity
@Table(name = "chat_histories", indexes = {
        @Index(name = "idx_chat_created_at", columnList = "createdAt")
})
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class ChatHistory extends BaseTimeEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** 사용자 질문 내용. */
    @Column(nullable = false, columnDefinition = "TEXT")
    private String question;

    /** AI 답변 내용. */
    @Column(nullable = false, columnDefinition = "TEXT")
    private String answer;

    /**
     * 답변의 주요 근거가 된 문서.
     * 참조 문서가 없는 답변(규정 외 질문 등)도 있을 수 있어 nullable 로 둔다.
     */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "document_id")
    private Document referencedDocument;

    /** 참조한 페이지 번호 목록 (예: "3,5,12"). */
    @Column(length = 255)
    private String referencedPages;

    /** 화면을 열 때 만든 대화 묶음 번호. 같은 대화의 질문들을 묶어 분석할 때 쓴다. */
    @Column(length = 36)
    private String conversationId;

    /** 멀티턴일 때 모델 서버가 이전 대화를 반영해 다시 쓴 질문. 멀티턴이 아니면 null. */
    @Column(columnDefinition = "TEXT")
    private String rewrittenQuestion;

    @Builder
    private ChatHistory(String question, String answer,
                        Document referencedDocument, String referencedPages,
                        String conversationId, String rewrittenQuestion) {
        this.question = question;
        this.answer = answer;
        this.referencedDocument = referencedDocument;
        this.referencedPages = referencedPages;
        this.conversationId = conversationId;
        this.rewrittenQuestion = rewrittenQuestion;
    }
}
