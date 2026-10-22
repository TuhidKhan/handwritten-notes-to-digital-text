/**
 * Frontend Controller for Handwritten Notes to Digital Text Converter.
 * 
 * Manages UI interactions, file uploads, sample selection, API communication,
 * side-by-side preview rendering, and text download.
 */

// DOM Elements
const statusIndicator = document.getElementById("statusIndicator");
const statusText = document.getElementById("statusText");
const settingsBtn = document.getElementById("settingsBtn");
const settingsModal = document.getElementById("settingsModal");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const resetBackendBtn = document.getElementById("resetBackendBtn");
const backendUrlInput = document.getElementById("backendUrlInput");
const reconnectBtn = document.getElementById("reconnectBtn");

const tabUpload = document.getElementById("tabUpload");
const tabSamples = document.getElementById("tabSamples");
const uploadView = document.getElementById("uploadView");
const samplesView = document.getElementById("samplesView");

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const sampleGrid = document.getElementById("sampleGrid");
const pipelineSelect = document.getElementById("pipelineSelect");
const extractBtn = document.getElementById("extractBtn");
const errorBanner = document.getElementById("errorBanner");

const origViewport = document.getElementById("origViewport");
const procViewport = document.getElementById("procViewport");
const origImg = document.getElementById("origImg");
const procImg = document.getElementById("procImg");

const metricsRow = document.getElementById("metricsRow");
const metricTime = document.getElementById("metricTime");
const metricConf = document.getElementById("metricConf");
const metricWords = document.getElementById("metricWords");

const groundTruthSection = document.getElementById("groundTruthSection");
const groundTruthText = document.getElementById("groundTruthText");
const extractedTextArea = document.getElementById("extractedTextArea");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusMessage = document.getElementById("statusMessage");

// Default Production Backend URL (Render Web Service)
const DEFAULT_RENDER_BACKEND = "https://handwritten-notes-to-digital-text.onrender.com";

function getDefaultBackendUrl() {
  const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  if (isLocal) {
    return "http://localhost:8000";
  }
  if (!window.location.host.includes("netlify.app")) {
    return window.location.origin;
  }
  return DEFAULT_RENDER_BACKEND;
}

function getBackendUrl() {
  const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  const saved = localStorage.getItem("ocr_backend_url");
  if (isLocal) {
    return (saved && (saved.includes("localhost") || saved.includes("127.0.0.1"))) ? saved : "http://localhost:8000";
  }
  if (!window.location.host.includes("netlify.app")) {
    return window.location.origin;
  }
  if (saved && !saved.includes("localhost") && !saved.includes("127.0.0.1") && !saved.includes("hf.space")) {
    return saved.trim().replace(/\/$/, "");
  }
  return DEFAULT_RENDER_BACKEND;
}

// Initialize input value
backendUrlInput.value = getBackendUrl();

// Application State
const state = {
  backendUrl: getBackendUrl(),
  selectedFile: null,
  selectedSampleId: null,
  selectedSampleGroundTruth: null,
  currentSourceName: "note",
  extractedText: "",
};

// -------------------------------------------------------------
// 1. Initialization and Backend Connection
// -------------------------------------------------------------

async function checkBackendHealth() {
  const url = getBackendUrl();
  state.backendUrl = url;
  statusIndicator.className = "status-indicator";
  statusText.textContent = "Connecting...";

  let wakeTimer = setTimeout(() => {
    statusText.textContent = "Waking server (30s)...";
  }, 3500);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    const response = await fetch(`${url}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    clearTimeout(wakeTimer);

    if (response.ok) {
      statusIndicator.className = "status-indicator online";
      statusText.textContent = "System Online";
      loadSamples();
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (err) {
    clearTimeout(wakeTimer);
    statusIndicator.className = "status-indicator offline";
    statusText.textContent = "Service Offline";
  }
}

// -------------------------------------------------------------
// 2. Sample Dataset Catalog Loading
// -------------------------------------------------------------

async function loadSamples() {
  const url = getBackendUrl();
  try {
    const res = await fetch(`${url}/api/samples`);
    if (!res.ok) throw new Error("Failed to fetch sample catalog");

    const data = await res.json();
    const samples = data.samples || [];

    if (samples.length === 0) {
      sampleGrid.innerHTML = '<div class="loading-placeholder">No IAM samples found.</div>';
      return;
    }

    sampleGrid.innerHTML = "";
    samples.forEach((sample) => {
      const item = document.createElement("div");
      item.className = "sample-item";
      item.dataset.sampleId = sample.id;

      item.innerHTML = `
        <img src="${sample.preview}" alt="${sample.id}" class="sample-thumbnail">
        <div class="sample-info">
          <div class="sample-id">${sample.id}</div>
          <div class="sample-text">${sample.ground_truth}</div>
        </div>
      `;

      item.addEventListener("click", () => selectSample(sample, item));
      sampleGrid.appendChild(item);
    });
  } catch (err) {
    sampleGrid.innerHTML = `<div class="loading-placeholder">Could not load samples: ${err.message}</div>`;
  }
}

function selectSample(sample, itemElement) {
  // Update state
  state.selectedFile = null;
  state.selectedSampleId = sample.id;
  state.selectedSampleGroundTruth = sample.ground_truth;
  state.currentSourceName = sample.id;

  // Highlight active UI item
  document.querySelectorAll(".sample-item").forEach((el) => el.classList.remove("selected"));
  itemElement.classList.add("selected");

  // Show immediate preview in original viewport
  showOriginalPreview(sample.preview);

  // Show ground truth
  groundTruthText.textContent = sample.ground_truth;
  groundTruthSection.style.display = "block";

  // Enable extract button
  extractBtn.disabled = false;
  clearResults();
}

// -------------------------------------------------------------
// 3. Tab Navigation (Upload vs Samples)
// -------------------------------------------------------------

tabUpload.addEventListener("click", () => {
  tabUpload.classList.add("active");
  tabSamples.classList.remove("active");
  uploadView.classList.add("active");
  samplesView.classList.remove("active");
});

tabSamples.addEventListener("click", () => {
  tabSamples.classList.add("active");
  tabUpload.classList.remove("active");
  samplesView.classList.add("active");
  uploadView.classList.remove("active");
});

// -------------------------------------------------------------
// 4. File Upload & Drag-and-Drop Handling
// -------------------------------------------------------------

browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    processSelectedFile(e.target.files[0]);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    processSelectedFile(e.dataTransfer.files[0]);
  }
});

function processSelectedFile(file) {
  const validExtensions = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"];
  if (!validExtensions.includes(file.type)) {
    showError("Please select a valid image file (PNG, JPG, JPEG, WEBP, BMP).");
    return;
  }

  const maxSize = 15 * 1024 * 1024;
  if (file.size > maxSize) {
    showError("File size exceeds 15 MB limit.");
    return;
  }

  hideError();
  state.selectedFile = file;
  state.selectedSampleId = null;
  state.selectedSampleGroundTruth = null;
  state.currentSourceName = file.name.replace(/\.[^/.]+$/, "");

  // Clear sample selection styling and ground truth
  document.querySelectorAll(".sample-item").forEach((el) => el.classList.remove("selected"));
  groundTruthSection.style.display = "none";

  // Preview local image immediately
  const reader = new FileReader();
  reader.onload = (event) => {
    showOriginalPreview(event.target.result);
  };
  reader.readAsDataURL(file);

  extractBtn.disabled = false;
  clearResults();
}

function showOriginalPreview(dataUrl) {
  origImg.src = dataUrl;
  origImg.style.display = "block";
  const placeholder = origViewport.querySelector(".viewport-placeholder");
  if (placeholder) placeholder.style.display = "none";
}

function clearResults() {
  procImg.style.display = "none";
  const procPlaceholder = procViewport.querySelector(".viewport-placeholder");
  if (procPlaceholder) {
    procPlaceholder.style.display = "block";
    procPlaceholder.textContent = "Awaiting extraction";
  }

  metricsRow.style.display = "none";
  extractedTextArea.value = "";
  copyBtn.disabled = true;
  downloadBtn.disabled = true;
  statusMessage.style.display = "none";
}

// -------------------------------------------------------------
// 5. OCR Extraction Execution
// -------------------------------------------------------------

extractBtn.addEventListener("click", async () => {
  if (!state.selectedFile && !state.selectedSampleId) {
    showError("Please select an input image first.");
    return;
  }

  hideError();
  extractBtn.disabled = true;
  extractBtn.textContent = "Processing and Extracting Text...";

  const procPlaceholder = procViewport.querySelector(".viewport-placeholder");
  if (procPlaceholder) {
    procPlaceholder.textContent = "Running OpenCV & EasyOCR...";
  }

  const formData = new FormData();
  formData.append("pipeline_mode", pipelineSelect.value);

  if (state.selectedFile) {
    formData.append("file", state.selectedFile);
  } else if (state.selectedSampleId) {
    formData.append("sample_id", state.selectedSampleId);
  }

  const url = getBackendUrl();

  try {
    const response = await fetch(`${url}/api/extract`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let errorMsg = `Server error (HTTP ${response.status})`;
      try {
        const errorData = await response.json();
        if (errorData.detail) errorMsg = errorData.detail;
      } catch (_) {}
      throw new Error(errorMsg);
    }

    const result = await response.json();
    renderExtractionSuccess(result);
  } catch (err) {
    showError(`Extraction failed: ${err.message}`);
    if (procPlaceholder) {
      procPlaceholder.textContent = "Extraction failed";
    }
  } finally {
    extractBtn.disabled = false;
    extractBtn.textContent = "Extract Handwritten Text";
  }
});

function renderExtractionSuccess(result) {
  // Render preprocessed image preview
  if (result.preprocessed_preview) {
    procImg.src = result.preprocessed_preview;
    procImg.style.display = "block";
    const placeholder = procViewport.querySelector(".viewport-placeholder");
    if (placeholder) placeholder.style.display = "none";
  }

  // Update metrics
  metricTime.textContent = `${result.processing_time_sec} s`;
  metricConf.textContent = `${Math.round(result.average_confidence * 100)}%`;
  metricWords.textContent = result.word_count;
  metricsRow.style.display = "grid";

  // Fill text area
  state.extractedText = result.extracted_text || "";
  extractedTextArea.value = state.extractedText;

  // Enable buttons if text exists
  copyBtn.disabled = !state.extractedText;
  downloadBtn.disabled = !state.extractedText;

  if (result.saved_file) {
    showNotice(`Extracted text saved locally to: ${result.saved_file}`);
  }
}

// -------------------------------------------------------------
// 6. Action Handlers: Copy and Download
// -------------------------------------------------------------

copyBtn.addEventListener("click", async () => {
  const text = extractedTextArea.value;
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
    showNotice("Extracted text copied to clipboard.");
  } catch (err) {
    // Fallback for older browsers
    extractedTextArea.select();
    document.execCommand("copy");
    showNotice("Extracted text copied to clipboard.");
  }
});

downloadBtn.addEventListener("click", () => {
  const text = extractedTextArea.value;
  if (!text) return;

  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);

  a.href = objectUrl;
  a.download = `extracted_${state.currentSourceName}_${timestamp}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(objectUrl);

  showNotice("Text file downloaded successfully.");
});

// -------------------------------------------------------------
// 7. Utility Feedback Helpers
// -------------------------------------------------------------

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.style.display = "block";
}

function hideError() {
  errorBanner.style.display = "none";
}

function showNotice(msg) {
  statusMessage.textContent = msg;
  statusMessage.style.display = "block";
  setTimeout(() => {
    statusMessage.style.display = "none";
  }, 4000);
}

// Settings Modal Events
if (settingsBtn && settingsModal) {
  settingsBtn.addEventListener("click", () => {
    backendUrlInput.value = getBackendUrl();
    settingsModal.style.display = "flex";
  });

  if (closeSettingsBtn) {
    closeSettingsBtn.addEventListener("click", () => {
      settingsModal.style.display = "none";
    });
  }

  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) {
      settingsModal.style.display = "none";
    }
  });

  if (resetBackendBtn) {
    resetBackendBtn.addEventListener("click", () => {
      localStorage.removeItem("ocr_backend_url");
      backendUrlInput.value = getDefaultBackendUrl();
      state.backendUrl = getDefaultBackendUrl();
      settingsModal.style.display = "none";
      checkBackendHealth();
      showNotice("Reset to default backend endpoint.");
    });
  }
}

if (reconnectBtn) {
  reconnectBtn.addEventListener("click", () => {
    const customUrl = backendUrlInput.value.trim().replace(/\/$/, "");
    if (customUrl) {
      localStorage.setItem("ocr_backend_url", customUrl);
      state.backendUrl = customUrl;
    }
    if (settingsModal) {
      settingsModal.style.display = "none";
    }
    checkBackendHealth();
    showNotice("Saved backend endpoint. Reconnecting...");
  });
}

// Initial boot check
checkBackendHealth();
