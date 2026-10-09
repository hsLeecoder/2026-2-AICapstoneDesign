package com.hsleecoder.ragchatbot.document.domain;

import com.hsleecoder.ragchatbot.global.entity.BaseTimeEntity;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 업로드된 PDF 문서의 메타데이터 엔티티.
 * 실제 본문 벡터는 Qdrant 에 저장되고, 여기서는 문서 정보와 처리 상태만 관리한다.
 * (업로드 시간은 BaseTimeEntity.createdAt 이 담당)
 */
@Getter
@Entity
@Table(name = "documents")
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Document extends BaseTimeEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** 사용자가 업로드한 원본 파일명. */
    @Column(nullable = false, length = 255)
    private String originalFileName;

    /** 서버 저장용 파일명 (UUID 등, 충돌 방지). */
    @Column(nullable = false, unique = true, length = 255)
    private String storedFileName;

    @Column(nullable = false, length = 100)
    private String contentType;

    /** 파일 크기 (byte). */
    @Column(nullable = false)
    private long fileSize;

    /** PDF 총 페이지 수 (파싱 후 채워짐). */
    private Integer pageCount;

    /** 이 문서에서 생성된 청크 수 (색인 후 채워짐). */
    private Integer chunkCount;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private DocumentStatus status;

    @Builder
    private Document(String originalFileName, String storedFileName,
                     String contentType, long fileSize) {
        this.originalFileName = originalFileName;
        this.storedFileName = storedFileName;
        this.contentType = contentType;
        this.fileSize = fileSize;
        this.status = DocumentStatus.UPLOADED;
    }

    /** 상태를 변경한다. 허용되지 않는 전이는 예외를 던진다. */
    public void changeStatus(DocumentStatus next) {
        if (!status.canTransitionTo(next)) {
            throw new IllegalStateException(
                    "허용되지 않는 상태 전이입니다: %s -> %s".formatted(status, next));
        }
        this.status = next;
    }

    /** 색인 완료 시 페이지/청크 수를 기록하고 상태를 INDEXED 로 변경한다. */
    public void markIndexed(int pageCount, int chunkCount) {
        changeStatus(DocumentStatus.INDEXED);
        this.pageCount = pageCount;
        this.chunkCount = chunkCount;
    }
}