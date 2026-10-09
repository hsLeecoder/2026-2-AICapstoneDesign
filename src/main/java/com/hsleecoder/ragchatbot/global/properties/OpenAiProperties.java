package com.hsleecoder.ragchatbot.global.properties;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * application.yaml 의 openai.* 설정을 바인딩하는 불변 record.
 */
@ConfigurationProperties(prefix = "openai")
public record OpenAiProperties(
        String apiKey,
        String baseUrl,
        String chatModel,
        String embeddingModel
) {
    /** API Key 설정 여부. 실제 호출 전에 검증하는 용도. */
    public boolean hasApiKey() {
        return apiKey != null && !apiKey.isBlank();
    }
}