package com.hsleecoder.ragchatbot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/**
 * 애플리케이션 진입점.
 * @ConfigurationPropertiesScan 으로 record 기반 설정 클래스(Qdrant, OpenAI)를 자동 등록한다.
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class RagChatbotApplication {

    public static void main(String[] args) {
        SpringApplication.run(RagChatbotApplication.class, args);
    }
}