package com.hsleecoder.ragchatbot.document.domain;

/**
 * PDF 문서의 처리 상태.
 * 업로드 → 처리 중(파싱/임베딩) → 색인 완료 / 실패 순서로 진행된다.
 */
public enum DocumentStatus {
    UPLOADED,
    PROCESSING,
    INDEXED,
    FAILED;

    /** 상태 전이 가능 여부를 switch 표현식으로 검증한다. */
    public boolean canTransitionTo(DocumentStatus next) {
        return switch (this) {
            case UPLOADED -> next == PROCESSING || next == FAILED;
            case PROCESSING -> next == INDEXED || next == FAILED;
            case FAILED -> next == PROCESSING;   // 재처리 허용
            case INDEXED -> false;               // 색인 완료 후에는 변경 불가
        };
    }
}