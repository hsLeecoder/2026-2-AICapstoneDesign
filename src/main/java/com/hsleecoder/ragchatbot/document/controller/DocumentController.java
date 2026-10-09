package com.hsleecoder.ragchatbot.document.controller;

import com.hsleecoder.ragchatbot.document.dto.DocumentResponse;
import com.hsleecoder.ragchatbot.document.service.DocumentService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/** 관리자용 문서 업로드/조회 API. */
@RestController
@RequestMapping("/api/admin/documents")
@RequiredArgsConstructor
public class DocumentController {

    private final DocumentService documentService;

    /** PDF 업로드. multipart/form-data 의 "file" 파트로 전달받는다. */
    @PostMapping
    public ResponseEntity<DocumentResponse> upload(@RequestParam("file") MultipartFile file) {
        DocumentResponse response = documentService.upload(file);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    /** 등록된 문서 목록 조회. */
    @GetMapping
    public ResponseEntity<List<DocumentResponse>> findAll() {
        return ResponseEntity.ok(documentService.findAll());
    }
}