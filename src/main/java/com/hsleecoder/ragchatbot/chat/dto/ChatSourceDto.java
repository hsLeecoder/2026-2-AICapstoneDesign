package com.hsleecoder.ragchatbot.chat.dto;

/**
 * 답변의 근거가 된 출처 정보.
 * content 는 근거로 쓴 조항 원문(공백 정리본)이고, 화면의 '출처 보기'에서 보여준다.
 */
public record ChatSourceDto(String fileName, int page, String article, double score, String content) {
}
