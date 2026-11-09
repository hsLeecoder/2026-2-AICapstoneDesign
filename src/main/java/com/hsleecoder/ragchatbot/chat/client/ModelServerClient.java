package com.hsleecoder.ragchatbot.chat.client;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.hsleecoder.ragchatbot.global.properties.ModelServerProperties;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.List;

/**
 * EC2 모델 서버(FastAPI)의 /answer 를 호출하는 클라이언트.
 * 검색 → 리랭크 → Qwen3-8B 답변 생성까지 모델 서버가 처리하고, 여기서는 결과만 받는다.
 */
@Component
public class ModelServerClient {

    private final WebClient webClient;
    private final ModelServerProperties props;

    public ModelServerClient(@Qualifier("modelServerWebClient") WebClient webClient,
                             ModelServerProperties props) {
        this.webClient = webClient;
        this.props = props;
    }

    /** 질문을 보내고 답변과 출처를 받는다. 연결 실패 시 예외가 발생한다. */
    public ModelAnswer answer(String question) {
        return webClient.post()
                .uri("/answer")
                .bodyValue(new AnswerRequest(question, props.topK()))
                .retrieve()
                .bodyToMono(ModelAnswer.class)
                .block(Duration.ofSeconds(props.timeoutSeconds()));
    }

    /** /answer 요청 본문. 모델 서버의 필드 이름(question, top_k)에 맞춘다. */
    public record AnswerRequest(String question, @JsonProperty("top_k") int topK) {
    }

    /** /answer 응답. 모르는 필드는 무시한다. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ModelAnswer(
            @JsonAlias({"response", "text", "output"}) String answer,
            List<ModelSource> sources
    ) {
    }

    /** 응답 안의 출처 하나. 필드 이름은 /search 결과(doc, page, heading, rerank_score)와 같다고 가정한다. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ModelSource(
            @JsonProperty("chunk_id") String chunkId,
            String doc,
            Integer page,
            String heading,
            @JsonProperty("rerank_score") Double rerankScore,
            Double score
    ) {
    }
}
