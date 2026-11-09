package com.hsleecoder.ragchatbot.chat.service;

import com.hsleecoder.ragchatbot.chat.client.ModelServerClient;
import com.hsleecoder.ragchatbot.chat.client.ModelServerClient.ModelAnswer;
import com.hsleecoder.ragchatbot.chat.client.ModelServerClient.ModelSource;
import com.hsleecoder.ragchatbot.chat.domain.ChatHistory;
import com.hsleecoder.ragchatbot.chat.dto.ChatRequest;
import com.hsleecoder.ragchatbot.chat.dto.ChatResponse;
import com.hsleecoder.ragchatbot.chat.dto.ChatSourceDto;
import com.hsleecoder.ragchatbot.chat.repository.ChatHistoryRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 질의응답을 담당하는 서비스.
 * 검색·리랭크·답변 생성은 EC2 모델 서버(/answer)가 처리하고,
 * 이 서비스는 호출 결과를 DTO 로 바꾸고 대화 이력을 MySQL 에 저장한다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ChatService {

    private final ChatHistoryRepository chatHistoryRepository;
    private final ModelServerClient modelServerClient;

    /** 질문을 모델 서버에 전달해 답변을 받고, 이력을 저장한다. */
    public ChatResponse ask(ChatRequest request) {
        String question = request.question();
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("질문 내용이 비어 있습니다.");
        }

        ModelAnswer result;
        try {
            result = modelServerClient.answer(question);
        } catch (Exception e) {
            log.error("모델 서버 호출 실패", e);
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                    "모델 서버에 연결할 수 없습니다. EC2 서버와 SSH 터널 상태를 확인하세요.");
        }
        if (result == null || result.answer() == null || result.answer().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "모델 서버가 빈 답변을 반환했습니다.");
        }
        log.debug("모델 서버 응답: answer={}, sources={}", result.answer(), result.sources());

        List<ChatSourceDto> sources = toSourceDtos(result.sources());
        String pages = sources.stream()
                .map(s -> String.valueOf(s.page()))
                .distinct()
                .collect(Collectors.joining(","));

        ChatHistory history = ChatHistory.builder()
                .question(question)
                .answer(result.answer())
                .referencedPages(pages.isBlank() ? null : pages)
                .build();
        chatHistoryRepository.save(history);

        return new ChatResponse(result.answer(), sources);
    }

    /** 모델 서버의 출처를 API 응답용 DTO 로 변환한다. 점수는 리랭커 점수를 우선 쓴다. */
    private List<ChatSourceDto> toSourceDtos(List<ModelSource> sources) {
        if (sources == null) {
            return List.of();
        }
        return sources.stream()
                .map(s -> new ChatSourceDto(
                        s.doc(),
                        s.page() == null ? 0 : s.page(),
                        s.heading(),
                        s.rerankScore() != null ? s.rerankScore()
                                : (s.score() == null ? 0.0 : s.score())))
                .toList();
    }
}
