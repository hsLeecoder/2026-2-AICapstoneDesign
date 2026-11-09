package com.hsleecoder.ragchatbot.global.properties;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * application.yaml 의 model-server.* 설정을 바인딩하는 불변 record.
 * EC2 에서 돌아가는 검색·답변 서버(FastAPI)의 접속 정보다.
 * SSH 터널을 쓰면 baseUrl 은 http://localhost:8000 이 된다.
 */
@ConfigurationProperties(prefix = "model-server")
public record ModelServerProperties(
        String baseUrl,
        Integer topK,
        Integer timeoutSeconds
) {
    /** 값이 비어 있으면 기본값으로 채운다. */
    public ModelServerProperties {
        if (baseUrl == null || baseUrl.isBlank()) {
            baseUrl = "http://localhost:8000";
        }
        if (topK == null) {
            topK = 3;
        }
        if (timeoutSeconds == null) {
            timeoutSeconds = 120;
        }
    }
}
