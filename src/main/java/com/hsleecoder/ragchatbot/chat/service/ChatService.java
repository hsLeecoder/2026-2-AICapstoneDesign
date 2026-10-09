package com.hsleecoder.ragchatbot.chat.service;

import com.hsleecoder.ragchatbot.chat.domain.ChatHistory;
import com.hsleecoder.ragchatbot.chat.dto.ChatRequest;
import com.hsleecoder.ragchatbot.chat.dto.ChatResponse;
import com.hsleecoder.ragchatbot.chat.repository.ChatHistoryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 질의응답을 담당하는 서비스.
 * TODO(다음 턴): 질문 임베딩 → Qdrant 검색 → 프롬프트 조립 → OpenAI 호출 로직으로 교체한다.
 * 지금은 배관(연결)이 정상 동작하는지 확인하기 위한 stub 응답만 반환한다.
 */
@Service
@RequiredArgsConstructor
public class ChatService {

    private final ChatHistoryRepository chatHistoryRepository;

    /** 질문을 받아 임시 답변을 생성하고 이력을 저장한다. */
    public ChatResponse ask(ChatRequest request) {
        String question = request.question();
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("질문 내용이 비어 있습니다.");
        }

        // TODO: 실제로는 여기서 임베딩 → Qdrant 검색 → LLM 호출이 이루어진다.
        String answer = "아직 문서 검색/답변 생성 파이프라인이 연결되지 않았습니다. (입력한 질문: \"%s\")"
                .formatted(question);

        ChatHistory history = ChatHistory.builder()
                .question(question)
                .answer(answer)
                .build();
        chatHistoryRepository.save(history);

        return new ChatResponse(answer, List.of());
    }
}