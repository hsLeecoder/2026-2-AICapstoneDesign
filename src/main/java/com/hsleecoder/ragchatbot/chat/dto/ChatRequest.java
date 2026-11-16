package com.hsleecoder.ragchatbot.chat.dto;

import java.util.List;

/**
 * 사용자 질문 요청 DTO.
 * history 는 같은 화면에서 이어진 이전 대화(오래된 것부터)이고, 없으면 첫 질문으로 본다.
 * conversationId 는 화면을 열 때 만든 대화 묶음 번호로, 이력을 묶어 분석하는 데만 쓴다.
 */
public record ChatRequest(String question, List<Turn> history, String conversationId) {

    /** 이전 대화 한 턴. */
    public record Turn(String question, String answer) {
    }
}
