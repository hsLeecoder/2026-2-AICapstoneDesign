package com.hsleecoder.ragchatbot.global.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/**
 * 모든 요청이 컨트롤러에 닿기 전에 접속 코드·관리자 키를 검사하는 필터.
 * 값은 .env 의 ACCESS_CODES, ADMIN_KEY 에서 읽는다 (판단 규칙은 AccessPolicy 참고).
 */
@Component
public class AccessFilter extends OncePerRequestFilter {

    private static final Logger log = LoggerFactory.getLogger(AccessFilter.class);

    private final AccessPolicy policy;

    public AccessFilter(@Value("${ACCESS_CODES:}") String accessCodes,
                        @Value("${ADMIN_KEY:}") String adminKey) {
        this.policy = new AccessPolicy(accessCodes, adminKey);
        log.info("[접근 제어] 챗봇 접속 코드 {}개 ({}) / 관리자 키 {}",
                policy.codeCount(),
                policy.isChatGateEnabled() ? "보호 켜짐" : "보호 꺼짐: 누구나 사용 가능",
                policy.hasAdminKey() ? "설정됨" : "미설정: 관리자 API 전체 차단");
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        // 컨테이너가 인코딩(%61 등)과 ../ 를 정리한 뒤의 경로를 사용한다
        String path = request.getServletPath();
        if (request.getPathInfo() != null) {
            path = path + request.getPathInfo();
        }

        AccessPolicy.Decision decision = policy.decide(
                request.getMethod(), path,
                request.getHeader("X-Access-Code"),
                request.getHeader("X-Admin-Key"));

        if (decision == AccessPolicy.Decision.ALLOW) {
            chain.doFilter(request, response);
            return;
        }

        String message = (decision == AccessPolicy.Decision.NEED_ADMIN_KEY)
                ? "관리자 키가 필요하거나 올바르지 않습니다."
                : "접속 코드가 필요하거나 올바르지 않습니다.";
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json;charset=UTF-8");
        response.getWriter().write("{\"error\":\"unauthorized\",\"message\":\"" + message + "\"}");
    }
}