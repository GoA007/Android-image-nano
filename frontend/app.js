const DEFAULT_API_PORT = "8000";

function inferApiBase() {
  const saved = localStorage.getItem("apiBase");
  if (saved) return saved;

  if (window.location.protocol.startsWith("http")) {
    return `${window.location.protocol}//${window.location.hostname}:${DEFAULT_API_PORT}`;
  }
  return `http://127.0.0.1:${DEFAULT_API_PORT}`;
}

const imageUpload = document.getElementById("imageUpload");
const targetInput = document.getElementById("target");
const promptInput = document.getElementById("prompt");
const apiBaseInput = document.getElementById("apiBase");
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("thresholdValue");

const startBtn = document.getElementById("startBtn");
const retryBtn = document.getElementById("retryBtn");
const saveBtn = document.getElementById("saveBtn");
const newBtn = document.getElementById("newBtn");

const progressEl = document.getElementById("progress");
const feedbackLoop = document.getElementById("feedbackLoop");
const feedbackText = document.getElementById("feedbackText");
const logsOutput = document.getElementById("logsOutput");

const originalPreview = document.getElementById("originalPreview");
const maskPreview = document.getElementById("maskPreview");
const resultPreview = document.getElementById("resultPreview");

let currentFile = null;
let latestResultDataUrl = "";
let pollTimer = null;
let apiBase = inferApiBase();

function showImage(img, src) {
  img.src = src;
  img.style.display = "block";
}

function setProgress(text, show = true) {
  progressEl.textContent = text;
  progressEl.style.display = show ? "block" : "none";
}

function setBusy(isBusy) {
  startBtn.disabled = isBusy;
  retryBtn.disabled = isBusy;
}

function normalizeApiBase(value) {
  const trimmed = (value || "").trim().replace(/\/+$/, "");
  if (!trimmed) return inferApiBase();
  return trimmed;
}

function resetForNewRun() {
  feedbackLoop.style.display = "none";
  setProgress("", false);
  resultPreview.style.display = "none";
  maskPreview.style.display = "none";
  latestResultDataUrl = "";
  saveBtn.disabled = true;
  logsOutput.value = "";
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
});

imageUpload.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  currentFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    showImage(originalPreview, e.target.result);
    resetForNewRun();
    setProgress("Image loaded. Click Start 3-Step Process.");
  };
  reader.readAsDataURL(file);
});

apiBaseInput.value = apiBase;
apiBaseInput.addEventListener("change", () => {
  apiBase = normalizeApiBase(apiBaseInput.value);
  apiBaseInput.value = apiBase;
  localStorage.setItem("apiBase", apiBase);
  setProgress(`Backend set to ${apiBase}`);
});

newBtn.addEventListener("click", () => {
  window.location.reload();
});

saveBtn.addEventListener("click", () => {
  if (!latestResultDataUrl) {
    alert("Generate an image first.");
    return;
  }
  const link = document.createElement("a");
  link.href = latestResultDataUrl;
  link.download = `kandinsky_edit_${Date.now()}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
});

startBtn.addEventListener("click", () => {
  initiateEdit("");
});

retryBtn.addEventListener("click", () => {
  initiateEdit(feedbackText.value.trim());
});

async function initiateEdit(feedback) {
  const prompt = promptInput.value.trim();
  const target = targetInput.value.trim() || "fingernails";
  const threshold = thresholdInput.value;

  if (!currentFile) {
    alert("Upload an image first.");
    return;
  }
  if (!prompt) {
    alert("Enter a prompt (e.g., silver metallic).");
    return;
  }

  stopPolling();
  setBusy(true);
  feedbackLoop.style.display = "none";
  apiBase = normalizeApiBase(apiBaseInput.value);
  apiBaseInput.value = apiBase;
  localStorage.setItem("apiBase", apiBase);
  setProgress("Uploading to pipeline...");
  logsOutput.value = "[client] Submitting job...\n";

  const formData = new FormData();
  formData.append("image", currentFile);
  formData.append("prompt", prompt);
  formData.append("target", target);
  formData.append("feedback", feedback || "");
  formData.append("detection_threshold", threshold);

  try {
    const response = await fetch(`${apiBase}/start-edit/`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok || !data.job_id) {
      throw new Error(data?.detail || "Failed to create job.");
    }
    logsOutput.value += `[client] Job accepted: ${data.job_id}\n`;

    pollTimer = setInterval(() => checkStatus(data.job_id), 1200);
  } catch (error) {
    setProgress(`Failed: ${error.message}`);
    setBusy(false);
  }
}

async function checkStatus(jobId) {
  try {
    const response = await fetch(`${apiBase}/status/${jobId}`);
    const data = await response.json();

    setProgress(data.message || "Running...");
    if (typeof data.logs_text === "string") {
      logsOutput.value = data.logs_text;
      logsOutput.scrollTop = logsOutput.scrollHeight;
    }

    if (data.stage === 4) {
      stopPolling();
      if (!data.result || !data.result.image_base64) {
        throw new Error("No result image was returned.");
      }

      latestResultDataUrl = `data:image/png;base64,${data.result.image_base64}`;
      showImage(resultPreview, latestResultDataUrl);
      if (data.result.mask_base64) {
        showImage(maskPreview, `data:image/png;base64,${data.result.mask_base64}`);
      }
      saveBtn.disabled = false;
      feedbackLoop.style.display = "block";
      setBusy(false);

      const blob = await (await fetch(latestResultDataUrl)).blob();
      currentFile = new File([blob], "edited_base.png", { type: "image/png" });
      return;
    }

    if (data.stage === -1) {
      stopPolling();
      throw new Error(data.message || "Pipeline failed.");
    }
  } catch (error) {
    stopPolling();
    setProgress(`Error: ${error.message}`);
    logsOutput.value += `[client] Error: ${error.message}\n`;
    setBusy(false);
  }
}
