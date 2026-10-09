package com.hsleecoder.ragchatbot.chat.dto;

import java.util.List;

/** 챗봇 응답 DTO. */
public record ChatResponse(String answer, List<ChatSourceDto> sources) {
}