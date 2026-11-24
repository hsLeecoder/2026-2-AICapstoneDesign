package com.hsleecoder.ragchatbot.global.security;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;

/**
 * 접근 규칙 판단기 (스프링과 무관한 순수 자바 클래스).
 * - /api/admin/** : 관리자 키(X-Admin-Key)가 맞아야 통과. 키가 설정되지 않았으면 전부 차단.
 * - 그 밖의 /api/** (챗봇 등) : 접속 코드(X-Access-Code)가 목록 중 하나와 같아야 통과.
 *   접속 코드를 하나도 설정하지 않았으면 보호를 끈 것으로 보고 통과(로컬 개발 편의).
 * - 화면 파일 등 /api 밖의 주소 : 항상 통과.
 */
public final class AccessPolicy {

    public enum Decision { ALLOW, NEED_ACCESS_CODE, NEED_ADMIN_KEY }

    private final Set<String> accessCodes;
    private final String adminKey;

    /** @param accessCodesCsv 쉼표로 구분한 접속 코드 목록 (예: "abc,def"). 비어 있으면 보호 꺼짐 */
    public AccessPolicy(String accessCodesCsv, String adminKey) {
        Set<String> codes = new LinkedHashSet<>();
        if (accessCodesCsv != null) {
            Arrays.stream(accessCodesCsv.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .forEach(codes::add);
        }
        this.accessCodes = codes;
        this.adminKey = adminKey == null ? "" : adminKey.trim();
    }

    public int codeCount() { return accessCodes.size(); }

    public boolean isChatGateEnabled() { return !accessCodes.isEmpty(); }

    public boolean hasAdminKey() { return !adminKey.isEmpty(); }

    /**
     * @param method 요청 방식 (GET, POST ...)
     * @param path   서블릿 컨테이너가 정리한 요청 경로 (예: /api/chat)
     */
    public Decision decide(String method, String path, String accessCodeHeader, String adminKeyHeader) {
        if ("OPTIONS".equalsIgnoreCase(method)) {
            return Decision.ALLOW; // 브라우저 사전 확인 요청은 통과
        }
        String p = normalize(path);
        if (!(p.equals("/api") || p.startsWith("/api/"))) {
            return Decision.ALLOW; // 화면 파일 등
        }
        if (p.equals("/api/admin") || p.startsWith("/api/admin/")) {
            if (!hasAdminKey() || !sameSecret(adminKeyHeader, adminKey)) {
                return Decision.NEED_ADMIN_KEY;
            }
            return Decision.ALLOW;
        }
        if (!isChatGateEnabled()) {
            return Decision.ALLOW;
        }
        for (String code : accessCodes) {
            if (sameSecret(accessCodeHeader, code)) {
                return Decision.ALLOW;
            }
        }
        return Decision.NEED_ACCESS_CODE;
    }

    /** 연속된 슬래시를 하나로 줄이고 소문자로 맞춘다 (우회 시도 방지용 이중 안전장치). */
    private static String normalize(String path) {
        if (path == null || path.isEmpty()) return "/";
        return path.replaceAll("/{2,}", "/").toLowerCase(Locale.ROOT);
    }

    /** 시간 차이로 값을 추측하지 못하게 하는 비교 */
    private static boolean sameSecret(String given, String expected) {
        if (given == null || expected == null) return false;
        return MessageDigest.isEqual(
                given.trim().getBytes(StandardCharsets.UTF_8),
                expected.getBytes(StandardCharsets.UTF_8));
    }
}