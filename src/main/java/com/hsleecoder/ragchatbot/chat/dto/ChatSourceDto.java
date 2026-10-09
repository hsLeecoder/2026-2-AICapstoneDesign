package com.hsleecoder.ragchatbot.chat.dto;

/**
 * 답변의 근거가 된 출처 정보.
 * 지금은 실제 검색 로직이 없어 항상 빈 리스트로 내려가지만,
 * 다음 턴에 Qdrant 검색 결과를 이 형태로 채워 넣을 예정이다.
 */
public record ChatSourceDto(String fileName, int page, String article, double score) {
}