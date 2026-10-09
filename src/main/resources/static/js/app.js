const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const documentList = document.getElementById("document-list");

const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const chatLog = document.getElementById("chat-log");

/** 문서 상태값을 한글 라벨로 변환한다. */
function statusLabel(status) {
  const labels = {
    UPLOADED: "업로드됨",
    PROCESSING: "처리 중",
    INDEXED: "색인 완료",
    FAILED: "실패",
  };
  return labels[status] ?? status;
}

/** 업로드된 문서 목록을 서버에서 받아와 화면에 그린다. */
async function loadDocuments() {
  try {
    const res = await fetch("/api/admin/documents");
    if (!res.ok) throw new Error("문서 목록 조회 실패");
    const documents = await res.json();

    documentList.innerHTML = "";
    if (documents.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "아직 업로드된 문서가 없습니다.";
      documentList.appendChild(li);
      return;
    }

    for (const doc of documents) {
      const li = document.createElement("li");

      const name = document.createElement("span");
      name.textContent = doc.originalFileName;

      const badge = document.createElement("span");
      badge.className = "doc-status";
      badge.textContent = statusLabel(doc.status);

      li.appendChild(name);
      li.appendChild(badge);
      documentList.appendChild(li);
    }
  } catch (err) {
    console.error(err);
  }
}

/** PDF 업로드 폼 제출 처리. */
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  uploadStatus.textContent = "업로드 중...";
  uploadStatus.className = "status-text";

  try {
    const res = await fetch("/api/admin/documents", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.message ?? "업로드에 실패했습니다.");
    }

    uploadStatus.textContent = "업로드 완료";
    uploadStatus.className = "status-text success";
    uploadForm.reset();
    await loadDocuments();
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.className = "status-text error";
  }
});

/** 채팅 말풍선을 추가한다. */
function appendBubble(text, sender) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${sender}`;
  bubble.textContent = text;
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

/** 질문 폼 제출 처리. */
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendBubble(question, "user");
  questionInput.value = "";
  questionInput.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) throw new Error("답변을 받아오지 못했습니다.");
    const data = await res.json();
    appendBubble(data.answer, "bot");
  } catch (err) {
    appendBubble(`오류: ${err.message}`, "bot");
  } finally {
    questionInput.disabled = false;
    questionInput.focus();
  }
});

loadDocuments();