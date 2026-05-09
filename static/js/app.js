let sessionId = null;

const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const chatHistory = document.getElementById("chatHistory");

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

const clearChatBtn = document.getElementById("clearChatBtn");
const newChatBtn = document.getElementById("newChatBtn");

// Append messages
function addMessage(text, sender) {
    const msg = document.createElement("div");
    msg.className = sender === "user" ? "user-msg" : "bot-msg";
    msg.innerText = text;
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Handle chat submit
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    let query = queryInput.value.trim();
    if (!query) return;

    addMessage(query, "user");
    queryInput.value = "";

    const formData = new FormData();
    formData.append("query", query);
    if (sessionId) formData.append("session_id", sessionId);

    addMessage("Typing...", "bot");

    const res = await fetch("/chat", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    // Remove typing
    chatHistory.removeChild(chatHistory.lastChild);

    sessionId = data.session_id;
    addMessage(data.answer, "bot");
});

// Upload file
uploadBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) {
        uploadStatus.innerText = "Select a file first!";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    uploadStatus.innerText = "Uploading...";

    const res = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    uploadStatus.innerText = data.message;
});

// Clear chat
clearChatBtn.addEventListener("click", () => {
    chatHistory.innerHTML = "";
});

// New Chat
newChatBtn.addEventListener("click", () => {
    sessionId = null;
    chatHistory.innerHTML = "";
    addMessage("🆕 New chat started.", "bot");
});
