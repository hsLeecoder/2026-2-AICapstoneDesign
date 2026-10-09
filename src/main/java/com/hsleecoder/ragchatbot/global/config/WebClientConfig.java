package com.hsleecoder.ragchatbot.global.config;

import com.hsleecoder.ragchatbot.global.properties.OpenAiProperties;
import com.hsleecoder.ragchatbot.global.properties.QdrantProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * 외부 API 통신용 WebClient 빈 설정.
 * 대상별로 baseUrl / 인증 헤더가 다르므로 빈을 분리하고, 이름(qdrantWebClient 등)으로 주입한다.
 */
@Configuration
public class WebClientConfig {

    /** Qdrant REST API 호출용 WebClient. */
    @Bean
    public WebClient qdrantWebClient(QdrantProperties props) {
        WebClient.Builder builder = WebClient.builder()
                .baseUrl(props.baseUrl())
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE);

        // 키가 설정된 경우에만 api-key 헤더 추가 (Qdrant Cloud 대응)
        if (props.hasApiKey()) {
            builder.defaultHeader("api-key", props.apiKey());
        }
        return builder.build();
    }

    /** OpenAI API 호출용 WebClient. */
    @Bean
    public WebClient openAiWebClient(OpenAiProperties props) {
        WebClient.Builder builder = WebClient.builder()
                .baseUrl(props.baseUrl())
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                // 임베딩 응답 등 큰 payload 대비 버퍼 확장 (10MB)
                .codecs(c -> c.defaultCodecs().maxInMemorySize(10 * 1024 * 1024));

        if (props.hasApiKey()) {
            builder.defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + props.apiKey());
        }
        return builder.build();
    }
}