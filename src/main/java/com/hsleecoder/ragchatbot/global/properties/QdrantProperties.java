package com.hsleecoder.ragchatbot.global.properties;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * application.yaml 의 qdrant.* 설정을 바인딩하는 불변 record.
 */
@ConfigurationProperties(prefix = "qdrant")
public record QdrantProperties(
        String host,
        int port,
        String apiKey,
        String collection,
        int vectorSize
) {
    /** Qdrant REST API 기본 URL 을 조합해 반환한다. */
    public String baseUrl() {
        return "http://%s:%d".formatted(host, port);
    }

    /** API Key 설정 여부 (로컬 Qdrant 는 키 없이 동작). */
    public boolean hasApiKey() {
        return apiKey != null && !apiKey.isBlank();
    }
}