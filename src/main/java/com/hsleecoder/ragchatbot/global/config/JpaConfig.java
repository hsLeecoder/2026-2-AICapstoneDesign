package com.hsleecoder.ragchatbot.global.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;

/**
 * JPA Auditing 활성화. BaseTimeEntity 의 생성/수정 시간 자동 기록에 필요하다.
 */
@Configuration
@EnableJpaAuditing
public class JpaConfig {
}