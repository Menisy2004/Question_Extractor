(() => {
  const form = document.querySelector("#upload-form");
  const input = document.querySelector("#pdf-file");
  const dropzone = document.querySelector("#dropzone");
  const dropTitle = document.querySelector("#drop-title");
  const selectedFile = document.querySelector("#selected-file");
  const generateButton = document.querySelector("#generate-button");
  const buttonLabel = generateButton.querySelector(".button-label");
  const errorMessage = document.querySelector("#error-message");
  const loadingState = document.querySelector("#loading-state");
  const resultsSection = document.querySelector("#results-section");
  const questionList = document.querySelector("#question-list");
  const copyButton = document.querySelector("#copy-button");
  let sessionId = null;
  let questions = [];

  function chooseFile(file) {
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      showError("That file does not look like a PDF. Please choose a PDF document.");
      input.value = "";
      return;
    }
    if (file.size > 30 * 1024 * 1024) {
      showError("This PDF is larger than 30 MB. Please choose a smaller document.");
      input.value = "";
      return;
    }
    clearError();
    selectedFile.textContent = `${file.name} · ${formatSize(file.size)}`;
    dropTitle.textContent = "Your document is ready";
    generateButton.disabled = false;
  }

  function formatSize(bytes) {
    return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
  }

  function clearError() {
    errorMessage.textContent = "";
    errorMessage.hidden = true;
  }

  input.addEventListener("change", () => chooseFile(input.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  }));
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
      chooseFile(file);
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!input.files[0]) return;
    clearError();
    loadingState.hidden = false;
    generateButton.disabled = true;
    buttonLabel.textContent = "Generating…";
    resultsSection.hidden = true;
    try {
      const response = await fetch("/generate", { method: "POST", body: new FormData(form) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Something went wrong while creating your questions.");
      sessionId = data.session_id;
      questions = data.questions;
      document.querySelector("#result-source").textContent = `Based on ${data.filename} · ${questions.length} questions`;
      renderQuestions();
      resultsSection.hidden = false;
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      showError(error.message || "Could not reach the question generator. Please try again.");
    } finally {
      loadingState.hidden = true;
      generateButton.disabled = !input.files[0];
      buttonLabel.textContent = "Generate questions";
    }
  });

  function renderQuestions() {
    questionList.replaceChildren();
    questions.forEach((question, index) => {
      const card = document.createElement("article");
      card.className = "question-card";
      const top = document.createElement("div");
      top.className = "question-top";
      const number = document.createElement("span");
      number.className = "question-number";
      number.textContent = String(index + 1).padStart(2, "0");
      const text = document.createElement("p");
      text.className = "question-text";
      text.textContent = question.replace(/^\s*\d+[.)]\s*/, "");
      const reveal = document.createElement("button");
      reveal.className = "reveal-button";
      reveal.type = "button";
      reveal.textContent = "Reveal answer +";
      top.append(number, text, reveal);
      card.append(top);
      reveal.addEventListener("click", () => revealAnswer(card, reveal, question.replace(/^\s*\d+[.)]\s*/, "")));
      questionList.append(card);
    });
  }

  async function revealAnswer(card, button, question) {
    const existing = card.querySelector(".answer-content");
    if (existing) {
      if (existing.classList.contains("answer-error")) {
        existing.remove();
      } else {
        existing.hidden = !existing.hidden;
        button.textContent = existing.hidden ? "Reveal answer +" : "Hide answer −";
        return;
      }
    }
    const answer = document.createElement("div");
    answer.className = "answer-content answer-loading";
    answer.textContent = "Looking through your material…";
    card.append(answer);
    button.disabled = true;
    button.textContent = "Finding answer…";
    try {
      const body = new FormData();
      body.set("session_id", sessionId);
      body.set("question", question);
      const response = await fetch("/answer", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not retrieve an answer.");
      answer.classList.remove("answer-loading");
      answer.textContent = data.answer;
      button.textContent = "Hide answer −";
    } catch (error) {
      answer.textContent = error.message;
      answer.classList.remove("answer-loading");
      answer.classList.add("answer-error");
      answer.style.color = "#a24d3c";
      button.textContent = "Try again +";
    } finally {
      button.disabled = false;
    }
  }

  copyButton.addEventListener("click", async () => {
    const text = questions.map((question, index) => `${index + 1}. ${question.replace(/^\s*\d+[.)]\s*/, "")}`).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      copyButton.querySelector("span").textContent = "Copied!";
      window.setTimeout(() => { copyButton.querySelector("span").textContent = "Copy questions"; }, 1600);
    } catch {
      showError("Clipboard access is unavailable in this browser.");
    }
  });
})();
