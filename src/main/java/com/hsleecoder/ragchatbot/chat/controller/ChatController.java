package com.hsleecoder.ragchatbot.chat.controller;

import com.hsleecoder.ragchatbot.chat.dto.ChatRequest;
import com.hsleecoder.ragchatbot.chat.dto.ChatResponse;
import com.hsleecoder.ragchatbot.chat.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** 사용자 질의응답 API. */
@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
public class ChatController {

    private final ChatService chatService;

    @PostMapping
    public ResponseEntity<ChatResponse> ask(@RequestBody ChatRequest request) {
        return ResponseEntity.ok(chatService.ask(request));
    }
}