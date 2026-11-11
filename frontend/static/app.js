/**
 * Fovea Studio Frontend Application (Vanilla ES6 Module)
 * Connects to the Fovea Backend API (/v1/* via /api/v1/*) without touching backend code.
 */

const ASPECT_CARDS = [
  { spec: "9:16", label: "9:16", desc: "Shorts / Reels / TikTok", w: 16, h: 28 },
  { spec: "1:1",  label: "1:1",  desc: "Square Feed",            w: 22, h: 22 },
  { spec: "4:5",  label: "4:5",  desc: "Portrait Social",        w: 20, h: 25 },
  { spec: "16:9", label: "16:9", desc: "Landscape Widescreen",   w: 28, h: 16 },
  { spec: "21:9", label: "21:9", desc: "Ultrawide Cinema",       w: 30, h: 13 },
];

const state = {
  userId: localStorage.getItem("fovea_user_id") || "user-default",
  geminiKey: localStorage.getItem("fovea_gemini_key") || "",
  theme: localStorage.getItem("fovea_theme") || "dark",
  config: {
    mode: "local",
    demo_mode: false,
    max_upload_mb: null,
    max_duration_s: null,
    daily_jobs_per_user: null,
    max_targets: 4,
    presets: {},
    options: [],
  },
  jobs: [],
  selectedJobId: null,
  selectedJobDetail: null,
  searchQuery: "",
  playerCollapsed: false,
  sidebarCollapsed: false,
  activePlayerTab: null, // "local" or output label like "9x16"
  localVideoBlobs: new Map(), // job_id -> blob URL for instant preview before/during processing
  transcripts: new Map(), // job_id -> array of transcript segments
  draftConfigs: new Map(), // job_id -> { targets: Set, customTarget: "", options: {...} }
  pollTimer: null,
};

// ============================================================================
// DOM References
// ============================================================================
const el = {
  sidebar: document.getElementById("sidebar"),
  collapseSidebarBtn: document.getElementById("collapse-sidebar-btn"),
  expandSidebarBtn: document.getElementById("expand-sidebar-btn"),
  newUploadBtn: document.getElementById("new-upload-btn"),
  searchInput: document.getElementById("job-search-input"),
  countActive: document.getElementById("count-active"),
  countUploaded: document.getElementById("count-uploaded"),
  countCompleted: document.getElementById("count-completed"),
  listActive: document.getElementById("list-active"),
  listUploaded: document.getElementById("list-uploaded"),
  listCompleted: document.getElementById("list-completed"),
  backendStatusDot: document.getElementById("backend-status-dot"),
  backendStatusText: document.getElementById("backend-status-text"),
  openSettingsBtn: document.getElementById("open-settings-btn"),
  currentUserLabel: document.getElementById("current-user-label"),

  workspaceVideoTitle: document.getElementById("workspace-video-title"),
  workspaceStatusBadge: document.getElementById("workspace-status-badge"),
  refreshJobsBtn: document.getElementById("refresh-jobs-btn"),
  toggleThemeBtn: document.getElementById("toggle-theme-btn"),
  togglePlayerBtn: document.getElementById("toggle-player-btn"),
  togglePlayerLabel: document.getElementById("toggle-player-label"),

  uploadView: document.getElementById("upload-view"),
  dropzoneCard: document.getElementById("dropzone-card"),
  fileInput: document.getElementById("file-input"),
  uploadLimitsBar: document.getElementById("upload-limits-bar"),
  uploadProgressBox: document.getElementById("upload-progress-box"),
  uploadFilenameLabel: document.getElementById("upload-filename-label"),
  uploadPctLabel: document.getElementById("upload-pct-label"),
  uploadProgressFill: document.getElementById("upload-progress-fill"),
  jobDetailView: document.getElementById("job-detail-view"),

  videoDrawer: document.getElementById("video-drawer"),
  collapsePlayerBtn: document.getElementById("collapse-player-btn"),
  playerFormatTabs: document.getElementById("player-format-tabs"),
  videoPlayer: document.getElementById("video-player"),
  playerEmptyState: document.getElementById("player-empty-state"),
  playerEmptyText: document.getElementById("player-empty-text"),
  playerTimeReadout: document.getElementById("player-time-readout"),
  stepBackBtn: document.getElementById("step-back-btn"),
  stepFwdBtn: document.getElementById("step-fwd-btn"),
  playbackRateSelect: document.getElementById("playback-rate-select"),
  downloadActiveVideoBtn: document.getElementById("download-active-video-btn"),
  downloadBtnLabel: document.getElementById("download-btn-label"),

  settingsModal: document.getElementById("settings-modal"),
  closeSettingsBtn: document.getElementById("close-settings-btn"),
  userIdInput: document.getElementById("user-id-input"),
  geminiKeyInput: document.getElementById("gemini-key-input"),
  saveSettingsBtn: document.getElementById("save-settings-btn"),
};

// ============================================================================
// Helpers
// ============================================================================
function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatStatusLabel(status) {
  switch (status) {
    case "awaiting_upload": return "Uploaded · Not Started";
    case "queued":          return "Queued";
    case "running":         return "Processing";
    case "rendered":        return "Videos Ready · Transcribing";
    case "complete":        return "Completed";
    case "failed":          return "Failed";
    default:                return status || "Unknown";
  }
}

function formatClock(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00.0";
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${String(m).padStart(2, "0")}:${s}`;
}

function formatDurationBetween(startIso, endIso) {
  if (!startIso) return "";
  const start = new Date(startIso).getTime();
  const end = endIso ? new Date(endIso).getTime() : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return "";
  const sec = ((end - start) / 1000).toFixed(1);
  return `${sec}s`;
}

function formatRelativeTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function getDraftConfig(jobId) {
  if (!state.draftConfigs.has(jobId)) {
    state.draftConfigs.set(jobId, {
      targets: new Set(["9:16"]),
      customTarget: "",
      options: {
        transcribe: false,
        focal_strategy: "viterbi",
        face_backend: "mediapipe",
        short_side: 1080,
        enable_speaker_detection: true,
        enable_ocr: true,
        enable_motion: true,
        debug_renders: false,
        pan_window_frames: 12,
        cut_threshold_px: 150,
        still_deadband_px: 18,
        detect_every: 2,
      },
    });
  }
  return state.draftConfigs.get(jobId);
}

// ============================================================================
// API Calls
// ============================================================================
async function checkHealthAndConfig() {
  try {
    const healthRes = await fetch("/api/health");
    const health = await healthRes.json();
    if (healthRes.ok && health.ok) {
      el.backendStatusDot.className = "conn-dot online";
      const mode = health.backend?.mode || "local";
      el.backendStatusText.textContent = `Online (${mode})`;
    } else {
      el.backendStatusDot.className = "conn-dot offline";
      el.backendStatusText.textContent = "Backend Offline";
    }
  } catch {
    el.backendStatusDot.className = "conn-dot offline";
    el.backendStatusText.textContent = "Backend Offline";
  }

  try {
    const cfgRes = await fetch("/api/v1/config");
    if (cfgRes.ok) {
      state.config = await cfgRes.json();
      renderLimitsBar();
    }
  } catch {
    // Keep defaults if backend is not yet started
  }
}

async function fetchJobsList() {
  try {
    const res = await fetch(`/api/v1/jobs?user_id=${encodeURIComponent(state.userId)}&limit=50`);
    if (!res.ok) return;
    const list = await res.json();
    // Filter out deleted jobs
    state.jobs = list.filter((j) => j.status !== "deleted");
    renderSidebarLists();

    if (state.selectedJobId) {
      await fetchSelectedJobDetail(state.selectedJobId);
    }
  } catch {
    // Ignore transient network errors during polling
  }
}

async function fetchSelectedJobDetail(jobId) {
  try {
    const res = await fetch(
      `/api/v1/jobs/${encodeURIComponent(jobId)}?user_id=${encodeURIComponent(state.userId)}`
    );
    if (!res.ok) return;
    const detail = await res.json();
    state.selectedJobDetail = detail;

    // Update the corresponding entry in state.jobs as well
    const idx = state.jobs.findIndex((j) => j.job_id === jobId);
    if (idx !== -1) {
      state.jobs[idx] = { ...state.jobs[idx], ...detail };
    } else {
      state.jobs.unshift(detail);
    }

    // If transcript_url just became available, fetch it once
    if (detail.transcript_url && !state.transcripts.has(jobId)) {
      fetchTranscript(jobId, detail.transcript_url);
    }

    renderSidebarLists();
    renderWorkspaceHeader();
    renderJobDetailView();
    syncVideoPlayerTabs();
  } catch {
    // Ignore transient errors
  }
}

async function fetchTranscript(jobId, url) {
  try {
    const res = await fetch(url);
    if (res.ok) {
      const segments = await res.json();
      state.transcripts.set(jobId, Array.isArray(segments) ? segments : []);
      if (state.selectedJobId === jobId) {
        renderJobDetailView();
      }
    }
  } catch {
    // Ignore
  }
}

// ============================================================================
// Upload Flow (Step 1: POST /v1/uploads -> Step 2: PUT upload.url)
// ============================================================================
async function handleFileUpload(file) {
  if (!file) return;

  // Client-side size check if demo_mode limit is set
  const maxMb = state.config.max_upload_mb;
  if (maxMb && file.size > maxMb * 1024 * 1024) {
    alert(`File is ${(file.size / (1024 * 1024)).toFixed(1)} MB, which exceeds the ${maxMb} MB limit.`);
    return;
  }

  el.uploadProgressBox.hidden = false;
  el.uploadFilenameLabel.textContent = `Uploading ${file.name}...`;
  el.uploadPctLabel.textContent = "0%";
  el.uploadProgressFill.style.width = "0%";

  try {
    // 1. Create upload slot & job record (status = "awaiting_upload")
    const createRes = await fetch("/api/v1/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.userId,
        filename: file.name,
        size_bytes: file.size || 1,
      }),
    });

    if (!createRes.ok) {
      const errData = await createRes.json().catch(() => ({}));
      throw new Error(errData.detail || `Failed to create upload (HTTP ${createRes.status})`);
    }

    const { job_id: jobId, upload } = await createRes.json();

    // Save local object URL so the user can immediately preview the source video in the right player
    const localBlobUrl = URL.createObjectURL(file);
    state.localVideoBlobs.set(jobId, localBlobUrl);

    // 2. Upload raw bytes directly to upload.url with upload.headers
    await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open(upload.method || "PUT", upload.url, true);

      const headers = upload.headers || {};
      Object.entries(headers).forEach(([k, v]) => {
        xhr.setRequestHeader(k, v);
      });

      xhr.upload.onprogress = (evt) => {
        if (evt.lengthComputable) {
          const pct = Math.round((evt.loaded / evt.total) * 100);
          el.uploadPctLabel.textContent = `${pct}%`;
          el.uploadProgressFill.style.width = `${pct}%`;
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve();
        } else {
          reject(new Error(`File upload failed (HTTP ${xhr.status}): ${xhr.responseText}`));
        }
      };

      xhr.onerror = () => reject(new Error("Network error while uploading file to storage link."));
      xhr.send(file);
    });

    el.uploadProgressBox.hidden = true;
    // Select the newly uploaded job (now in "Uploaded · Not Started" state)
    await fetchJobsList();
    selectJob(jobId);
  } catch (err) {
    el.uploadFilenameLabel.textContent = `Upload failed: ${err.message}`;
    el.uploadPctLabel.textContent = "Error";
  }
}

// ============================================================================
// Start Processing Flow (POST /v1/jobs/{job_id}/start)
// ============================================================================
async function startJobProcessing(jobId) {
  const draft = getDraftConfig(jobId);
  const targets = Array.from(draft.targets);
  if (draft.customTarget.trim()) {
    targets.push(draft.customTarget.trim());
  }

  if (targets.length === 0) {
    alert("Select at least one output aspect ratio before starting.");
    return;
  }
  const maxTargets = state.config.max_targets || 4;
  if (targets.length > maxTargets) {
    alert(`Please select at most ${maxTargets} output formats.`);
    return;
  }

  const res = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: state.userId,
      targets,
      options: draft.options,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(err.detail || `Could not start job (HTTP ${res.status})`);
    return;
  }

  await fetchSelectedJobDetail(jobId);
  await fetchJobsList();
}

async function deleteJob(jobId, evt) {
  if (evt) evt.stopPropagation();
  if (!confirm("Delete this video job and its generated files?")) return;

  const res = await fetch(
    `/api/v1/jobs/${encodeURIComponent(jobId)}?user_id=${encodeURIComponent(state.userId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(err.detail || "Could not delete job");
    return;
  }

  if (state.selectedJobId === jobId) {
    state.selectedJobId = null;
    state.selectedJobDetail = null;
    showUploadView();
  }
  await fetchJobsList();
}

// ============================================================================
// UI Rendering: Sidebar
// ============================================================================
function renderLimitsBar() {
  const parts = [];
  if (state.config.max_upload_mb) parts.push(`Max size: ${state.config.max_upload_mb} MB`);
  if (state.config.max_duration_s) parts.push(`Max length: ${state.config.max_duration_s}s`);
  if (state.config.daily_jobs_per_user) parts.push(`Daily quota: ${state.config.daily_jobs_per_user} videos/day`);
  if (parts.length === 0) {
    parts.push(`Up to ${state.config.max_targets || 4} simultaneous aspect ratios per run`);
  }
  el.uploadLimitsBar.innerHTML = parts.map((p) => `<span>• ${escapeHtml(p)}</span>`).join("");
}

function renderSidebarLists() {
  const q = state.searchQuery.trim().toLowerCase();
  const filtered = state.jobs.filter((j) => {
    if (!q) return true;
    return (j.filename || "").toLowerCase().includes(q) || (j.job_id || "").toLowerCase().includes(q);
  });

  const activeJobs = filtered.filter((j) =>
    ["queued", "running", "rendered"].includes(j.status)
  );
  const uploadedJobs = filtered.filter((j) => j.status === "awaiting_upload");
  const completedJobs = filtered.filter((j) =>
    ["complete", "failed"].includes(j.status)
  );

  el.countActive.textContent = String(activeJobs.length);
  el.countUploaded.textContent = String(uploadedJobs.length);
  el.countCompleted.textContent = String(completedJobs.length);

  el.listActive.innerHTML = activeJobs.length
    ? activeJobs.map(renderSidebarJobItem).join("")
    : `<div class="sidebar-empty">No videos currently processing</div>`;

  el.listUploaded.innerHTML = uploadedJobs.length
    ? uploadedJobs.map(renderSidebarJobItem).join("")
    : `<div class="sidebar-empty">No queued uploads waiting to start</div>`;

  el.listCompleted.innerHTML = completedJobs.length
    ? completedJobs.map(renderSidebarJobItem).join("")
    : `<div class="sidebar-empty">No processed videos yet</div>`;

  // Bind click listeners
  document.querySelectorAll(".job-item[data-job-id]").forEach((btn) => {
    btn.addEventListener("click", () => selectJob(btn.dataset.jobId));
  });
  document.querySelectorAll(".job-delete-btn[data-delete-id]").forEach((btn) => {
    btn.addEventListener("click", (e) => deleteJob(btn.dataset.deleteId, e));
  });
}

function renderSidebarJobItem(job) {
  const isSelected = job.job_id === state.selectedJobId;
  const pct = Math.round((job.progress || 0) * 100);
  const targets = job.targets || [];
  const canDelete = !["queued", "running", "rendered"].includes(job.status);

  let metaHtml = "";
  if (job.status === "awaiting_upload") {
    metaHtml = `<span>Ready to configure</span>`;
  } else if (job.status === "queued") {
    metaHtml = `<span>Queued ${job.queue_position ? `(#${job.queue_position})` : ""}</span>`;
  } else if (job.status === "running" || job.status === "rendered") {
    metaHtml = `<span>${pct}%</span>`;
  } else if (job.status === "failed") {
    metaHtml = `<span style="color: var(--status-failed);">Failed</span>`;
  } else {
    metaHtml = `<span>${formatRelativeTime(job.finished_at || job.created_at)}</span>`;
  }

  const targetBadges = targets
    .map((t) => `<span class="mini-tag">${escapeHtml(t)}</span>`)
    .join("");

  const progressBar =
    job.status === "running" || job.status === "queued" || job.status === "rendered"
      ? `<div class="job-item-progress"><div class="job-item-progress-bar" style="width: ${Math.max(4, pct)}%"></div></div>`
      : "";

  return `
    <div class="job-item ${isSelected ? "active" : ""}" data-job-id="${escapeHtml(job.job_id)}" role="button" tabindex="0">
      <span class="job-status-dot ${escapeHtml(job.status)}"></span>
      <div class="job-item-body">
        <div class="job-item-title" title="${escapeHtml(job.filename || job.job_id)}">
          ${escapeHtml(job.filename || job.job_id)}
        </div>
        <div class="job-item-meta">
          ${metaHtml}
          ${targetBadges}
        </div>
        ${progressBar}
      </div>
      ${
        canDelete
          ? `<button class="job-delete-btn" data-delete-id="${escapeHtml(job.job_id)}" title="Delete job" aria-label="Delete job">✕</button>`
          : ""
      }
    </div>
  `;
}

// ============================================================================
// UI Rendering: Center Workspace (Header + Config + Process Status)
// ============================================================================
function showUploadView() {
  state.selectedJobId = null;
  state.selectedJobDetail = null;
  el.uploadView.hidden = false;
  el.jobDetailView.hidden = true;
  el.workspaceVideoTitle.textContent = "Upload a Video";
  el.workspaceStatusBadge.hidden = true;
  renderSidebarLists();
  syncVideoPlayerTabs();
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  state.activePlayerTab = null;
  el.uploadView.hidden = true;
  el.jobDetailView.hidden = false;
  await fetchSelectedJobDetail(jobId);
}

function renderWorkspaceHeader() {
  const job = state.selectedJobDetail;
  if (!job) {
    el.workspaceVideoTitle.textContent = "Upload a Video";
    el.workspaceStatusBadge.hidden = true;
    return;
  }
  el.workspaceVideoTitle.textContent = job.filename || job.job_id;
  el.workspaceStatusBadge.hidden = false;
  el.workspaceStatusBadge.className = `status-badge ${job.status}`;
  el.workspaceStatusBadge.textContent = formatStatusLabel(job.status);
}

function renderJobDetailView() {
  const job = state.selectedJobDetail;
  if (!job) return;

  const parts = [];

  // 1. If job is uploaded but hasn't started processing yet ("awaiting_upload"), show Configuration Panel
  if (job.status === "awaiting_upload") {
    parts.push(renderConfigureAndStartPanel(job));
  } else {
    // 2. Overall Process Progress Banner
    parts.push(renderOverallProgressBanner(job));
  }

  // 3. Always show the 4-Step Pipeline Breakdown so the user sees what will run / is running / ran
  parts.push(renderPipelineStepsTimeline(job));

  el.jobDetailView.innerHTML = parts.join("");
  bindJobDetailEvents(job);
}

function renderConfigureAndStartPanel(job) {
  const draft = getDraftConfig(job.job_id);
  const ratioCardsHtml = ASPECT_CARDS.map((card) => {
    const isSelected = draft.targets.has(card.spec);
    return `
      <button type="button" class="ratio-card ${isSelected ? "selected" : ""}" data-ratio-spec="${escapeHtml(card.spec)}">
        <div class="ratio-frame-preview" style="width: ${card.w}px; height: ${card.h}px;"></div>
        <div class="ratio-label">${escapeHtml(card.label)}</div>
        <div class="ratio-desc">${escapeHtml(card.desc)}</div>
      </button>
    `;
  }).join("");

  const opts = draft.options;

  return `
    <div class="panel-card" style="margin-bottom: 18px;">
      <div class="panel-card-header">
        <div>
          <div class="panel-card-title">Configure & Start Processing</div>
          <div class="panel-card-subtitle">
            This video is uploaded and ready. Pick up to ${state.config.max_targets || 4} output aspect ratios and launch the Fovea pipeline.
          </div>
        </div>
        <span class="mini-tag">Job ID: ${escapeHtml(job.job_id)}</span>
      </div>

      <div class="ratio-grid">
        ${ratioCardsHtml}
      </div>

      <div class="custom-target-row">
        <label for="custom-target-input" class="option-label" style="white-space: nowrap;">Custom Ratio / Size:</label>
        <input
          id="custom-target-input"
          type="text"
          class="text-input"
          style="max-width: 180px;"
          placeholder="e.g. 2.35:1 or 720x1280"
          value="${escapeHtml(draft.customTarget)}"
        />
        <span style="font-size: 12px; color: var(--text-muted);">Analysis runs once regardless of how many formats you pick.</span>
      </div>

      <div class="options-grid">
        <div class="option-field">
          <label class="option-label" for="opt-focal-strategy">Framing Strategy</label>
          <select id="opt-focal-strategy" class="select-input">
            <option value="viterbi" ${opts.focal_strategy === "viterbi" ? "selected" : ""}>Viterbi Global Optimization (Default)</option>
            <option value="rules" ${opts.focal_strategy === "rules" ? "selected" : ""}>Rule-Based Priority</option>
          </select>
        </div>

        <div class="option-field">
          <label class="option-label" for="opt-face-backend">Face / Speaker Backend</label>
          <select id="opt-face-backend" class="select-input">
            <option value="mediapipe" ${opts.face_backend === "mediapipe" ? "selected" : ""}>MediaPipe BlazeFace (Default)</option>
            <option value="pose" ${opts.face_backend === "pose" ? "selected" : ""}>YOLO-Pose GPU Keypoints</option>
          </select>
        </div>

        <div class="option-field">
          <label class="option-label" for="opt-short-side">Output Short Side (px)</label>
          <select id="opt-short-side" class="select-input">
            <option value="1080" ${Number(opts.short_side) === 1080 ? "selected" : ""}>1080p (Full HD)</option>
            <option value="720" ${Number(opts.short_side) === 720 ? "selected" : ""}>720p (HD Fast)</option>
          </select>
        </div>

        <div class="option-field">
          <label class="option-label" for="opt-detect-every">Detector Interval (frames)</label>
          <select id="opt-detect-every" class="select-input">
            <option value="1" ${Number(opts.detect_every) === 1 ? "selected" : ""}>Every frame (1)</option>
            <option value="2" ${Number(opts.detect_every) === 2 ? "selected" : ""}>Every 2 frames (Default)</option>
            <option value="4" ${Number(opts.detect_every) === 4 ? "selected" : ""}>Every 4 frames (Faster)</option>
          </select>
        </div>
      </div>

      <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 18px;">
        <label class="checkbox-row">
          <input id="opt-transcribe" type="checkbox" ${opts.transcribe ? "checked" : ""} />
          <span>Generate Whisper Transcript (Demucs + faster-whisper)</span>
        </label>
        <label class="checkbox-row">
          <input id="opt-speakers" type="checkbox" ${opts.enable_speaker_detection ? "checked" : ""} />
          <span>Active Speaker Detection (TalkNet)</span>
        </label>
        <label class="checkbox-row">
          <input id="opt-ocr" type="checkbox" ${opts.enable_ocr ? "checked" : ""} />
          <span>On-Screen Text Detection (EasyOCR)</span>
        </label>
        <label class="checkbox-row">
          <input id="opt-motion" type="checkbox" ${opts.enable_motion ? "checked" : ""} />
          <span>Optical Flow Motion</span>
        </label>
      </div>

      <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
        <button id="start-job-btn" type="button" class="primary-action-btn">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
          <span>Start Processing Video</span>
        </button>
        <span style="font-size: 12px; color: var(--text-muted);">
          Selected targets: <strong>${escapeHtml(Array.from(draft.targets).join(", ") || "None")}</strong>
        </span>
      </div>
    </div>
  `;
}

function renderOverallProgressBanner(job) {
  const pct = Math.round((job.progress || 0) * 100);
  const targets = (job.targets || []).map((t) => `<span class="mini-tag">${escapeHtml(t)}</span>`).join(" ");
  const elapsed = formatDurationBetween(job.started_at || job.queued_at, job.finished_at);

  return `
    <div class="overall-status-banner" style="margin-bottom: 18px;">
      <div class="overall-top-row">
        <div>
          <div style="font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 10px;">
            <span>Pipeline Execution Status</span>
            ${targets}
          </div>
          <div style="font-size: 12.5px; color: var(--text-secondary); margin-top: 3px;">
            ${
              job.status === "queued"
                ? `Waiting in queue${job.queue_position ? ` (Position #${job.queue_position})` : ""}...`
                : job.status === "running"
                ? `Running active step on worker...`
                : job.status === "rendered"
                ? `Reframed videos are ready! Finishing audio transcription in parallel...`
                : job.status === "complete"
                ? `All pipeline steps finished.`
                : job.status === "failed"
                ? `Pipeline stopped with an error.`
                : ""
            }
          </div>
        </div>
        <div style="text-align: right;">
          <div class="overall-pct">${pct}%</div>
          ${elapsed ? `<div style="font-size: 11.5px; color: var(--text-muted);">Elapsed: ${escapeHtml(elapsed)}</div>` : ""}
        </div>
      </div>

      <div class="progress-track">
        <div
          class="progress-fill"
          style="width: ${pct}%; background-color: ${
            job.status === "failed"
              ? "var(--status-failed)"
              : job.status === "complete" || job.status === "rendered"
              ? "var(--status-complete)"
              : "var(--status-running)"
          };"
        ></div>
      </div>

      ${job.error ? `<div class="error-box">${escapeHtml(job.error)}</div>` : ""}
    </div>
  `;
}

function renderPipelineStepsTimeline(job) {
  const st = job.steps || {};
  const prep = st.prep || {};
  const visual = st.visual || {};
  const audio = st.audio || {};
  const render = st.render || {};

  // Visual shards breakdown
  const doneShards = Array.isArray(visual.done) ? visual.done : [];
  const totalShards = Number(visual.total) || (doneShards.length > 0 ? doneShards.length : 1);
  const shardPills = [];
  for (let i = 0; i < totalShards; i++) {
    const isDone = doneShards.includes(i) || visual.status === "done";
    const isRunning = !isDone && visual.status === "running";
    shardPills.push(`
      <span class="shard-pill ${isDone ? "done" : isRunning ? "running" : ""}">
        <span>Shard ${i + 1}/${totalShards}</span>
        <span>${isDone ? "✓ Done" : isRunning ? "Running..." : "Pending"}</span>
      </span>
    `);
  }

  // Output format chips in Step 4 (render)
  const outputEntries = Object.entries(job.outputs || {});
  const outputsHtml = outputEntries.length
    ? `<div class="outputs-row">
        ${outputEntries
          .map(
            ([label, url]) => `
            <div class="output-chip">
              <span class="output-chip-label">${escapeHtml(label)}</span>
              <button type="button" class="secondary-btn watch-output-btn" data-output-label="${escapeHtml(label)}">
                ▶ Watch in Player
              </button>
              <a href="${escapeHtml(url)}" download="fovea_${escapeHtml(label)}.mp4" class="secondary-btn" style="text-decoration: none;">
                ↓ Download
              </a>
            </div>
          `
          )
          .join("")}
      </div>`
    : "";

  // Interactive transcript in Step 3 (audio)
  const transcriptSegs = state.transcripts.get(job.job_id);
  let transcriptHtml = "";
  if (transcriptSegs && transcriptSegs.length > 0) {
    transcriptHtml = `
      <div style="margin-top: 10px; font-weight: 500; color: var(--text-primary); font-size: 12.5px;">
        Interactive Transcript (click any segment to jump in the video player):
      </div>
      <div class="transcript-box">
        ${transcriptSegs
          .map(
            (seg) => `
            <button type="button" class="transcript-seg" data-seek-sec="${Number(seg.start_s || 0)}">
              <span class="transcript-time">${formatClock(Number(seg.start_s || 0))}</span>
              <span class="transcript-text">${escapeHtml(seg.segment_transcript || "")}</span>
            </button>
          `
          )
          .join("")}
      </div>
    `;
  } else if (audio.status === "done" && job.transcript_url) {
    transcriptHtml = `<div style="margin-top: 8px;">Transcript generated (<a href="${escapeHtml(job.transcript_url)}" target="_blank" style="color: var(--accent-primary);">open JSON</a>).</div>`;
  }

  return `
    <div class="steps-timeline">
      <!-- STEP 1: PREP -->
      <div class="step-card ${escapeHtml(prep.status || "")}">
        <div class="step-header">
          <div class="step-left">
            <div class="step-icon">${prep.status === "done" ? "✓" : "1"}</div>
            <div>
              <div class="step-title">Step 1 · Video Prep & Shot Detection</div>
              <div class="step-desc">720p@25fps proxy encode, 16 kHz audio extraction, and PySceneDetect shot boundaries</div>
            </div>
          </div>
          <div class="step-meta">
            ${escapeHtml((prep.status || "pending").toUpperCase())}
            ${prep.started_at ? ` · ${formatDurationBetween(prep.started_at, prep.finished_at)}` : ""}
          </div>
        </div>
        ${prep.error ? `<div class="error-box">${escapeHtml(prep.error)}</div>` : ""}
      </div>

      <!-- STEP 2: VISUAL -->
      <div class="step-card ${escapeHtml(visual.status || "")}">
        <div class="step-header">
          <div class="step-left">
            <div class="step-icon">${visual.status === "done" ? "✓" : "2"}</div>
            <div>
              <div class="step-title">Step 2 · Visual Analysis (Tracking, Speaker ASD, OCR & Motion)</div>
              <div class="step-desc">YOLO object tracking, BlazeFace/TalkNet active speaker scoring, EasyOCR text boxes, optical flow</div>
            </div>
          </div>
          <div class="step-meta">
            ${escapeHtml((visual.status || "pending").toUpperCase())}
            · ${doneShards.length}/${totalShards} shard(s)
            ${visual.started_at ? ` · ${formatDurationBetween(visual.started_at, visual.finished_at)}` : ""}
          </div>
        </div>
        <div class="step-body">
          <div>Parallel Shot Shards:</div>
          <div class="shards-grid">${shardPills.join("")}</div>
          ${visual.error ? `<div class="error-box">${escapeHtml(visual.error)}</div>` : ""}
        </div>
      </div>

      <!-- STEP 3: AUDIO -->
      <div class="step-card ${escapeHtml(audio.status || "")}">
        <div class="step-header">
          <div class="step-left">
            <div class="step-icon">${audio.status === "done" ? "✓" : "3"}</div>
            <div>
              <div class="step-title">Step 3 · Audio Vocal Isolation & Whisper Transcription (Parallel)</div>
              <div class="step-desc">htdemucs vocal stem separation + faster-whisper batched word-level transcription</div>
            </div>
          </div>
          <div class="step-meta">
            ${escapeHtml((audio.status || "optional / idle").toUpperCase())}
            ${audio.started_at ? ` · ${formatDurationBetween(audio.started_at, audio.finished_at)}` : ""}
          </div>
        </div>
        ${
          transcriptHtml || audio.error
            ? `<div class="step-body">
                ${transcriptHtml}
                ${audio.error ? `<div class="error-box">${escapeHtml(audio.error)}</div>` : ""}
              </div>`
            : ""
        }
      </div>

      <!-- STEP 4: RENDER -->
      <div class="step-card ${escapeHtml(render.status || "")}">
        <div class="step-header">
          <div class="step-left">
            <div class="step-icon">${render.status === "done" ? "✓" : "4"}</div>
            <div>
              <div class="step-title">Step 4 · Camera Path Planning & Multi-Format Render</div>
              <div class="step-desc">Viterbi focal point optimization, hold/pan/cut trajectory smoothing, and high-res FFmpeg encoding</div>
            </div>
          </div>
          <div class="step-meta">
            ${escapeHtml((render.status || "pending").toUpperCase())}
            ${render.started_at ? ` · ${formatDurationBetween(render.started_at, render.finished_at)}` : ""}
          </div>
        </div>
        ${
          outputsHtml || render.error
            ? `<div class="step-body">
                <div>Rendered Outputs:</div>
                ${outputsHtml}
                ${render.error ? `<div class="error-box">${escapeHtml(render.error)}</div>` : ""}
              </div>`
            : ""
        }
      </div>
    </div>
  `;
}

function bindJobDetailEvents(job) {
  const draft = getDraftConfig(job.job_id);

  // Aspect ratio card toggles
  document.querySelectorAll(".ratio-card[data-ratio-spec]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const spec = btn.dataset.ratioSpec;
      if (draft.targets.has(spec)) {
        if (draft.targets.size > 1) draft.targets.delete(spec);
      } else {
        if (draft.targets.size < (state.config.max_targets || 4)) {
          draft.targets.add(spec);
        }
      }
      renderJobDetailView();
    });
  });

  const customInput = document.getElementById("custom-target-input");
  if (customInput) {
    customInput.addEventListener("input", (e) => {
      draft.customTarget = e.target.value;
    });
  }

  const bindSelect = (id, key, isNum = false) => {
    const node = document.getElementById(id);
    if (node) {
      node.addEventListener("change", (e) => {
        draft.options[key] = isNum ? Number(e.target.value) : e.target.value;
      });
    }
  };
  bindSelect("opt-focal-strategy", "focal_strategy", false);
  bindSelect("opt-face-backend", "face_backend", false);
  bindSelect("opt-short-side", "short_side", true);
  bindSelect("opt-detect-every", "detect_every", true);

  const bindCheck = (id, key) => {
    const node = document.getElementById(id);
    if (node) {
      node.addEventListener("change", (e) => {
        draft.options[key] = Boolean(e.target.checked);
      });
    }
  };
  bindCheck("opt-transcribe", "transcribe");
  bindCheck("opt-speakers", "enable_speaker_detection");
  bindCheck("opt-ocr", "enable_ocr");
  bindCheck("opt-motion", "enable_motion");

  const startBtn = document.getElementById("start-job-btn");
  if (startBtn) {
    startBtn.addEventListener("click", () => startJobProcessing(job.job_id));
  }

  // Output watch buttons
  document.querySelectorAll(".watch-output-btn[data-output-label]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (state.playerCollapsed) togglePlayerDrawer(false);
      switchPlayerTab(btn.dataset.outputLabel);
    });
  });

  // Transcript click-to-seek buttons
  document.querySelectorAll(".transcript-seg[data-seek-sec]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const sec = Number(btn.dataset.seekSec || 0);
      if (state.playerCollapsed) togglePlayerDrawer(false);
      if (!el.videoPlayer.hidden) {
        el.videoPlayer.currentTime = sec;
        el.videoPlayer.play().catch(() => {});
      }
    });
  });
}

// ============================================================================
// UI Rendering: Right Collapsible Video Player Drawer
// ============================================================================
function syncVideoPlayerTabs() {
  const job = state.selectedJobDetail;
  if (!job) {
    el.playerFormatTabs.innerHTML = "";
    el.videoPlayer.hidden = true;
    el.videoPlayer.removeAttribute("src");
    el.playerEmptyState.hidden = false;
    el.playerEmptyText.textContent = "Select a video from the sidebar to preview the source or processed outputs.";
    el.downloadActiveVideoBtn.hidden = true;
    return;
  }

  const tabs = [];
  const localBlob = state.localVideoBlobs.get(job.job_id);
  if (localBlob) {
    tabs.push({ id: "local", label: "Original (Local)", url: localBlob, isDownloadable: false });
  }

  const outputs = job.outputs || {};
  Object.entries(outputs).forEach(([label, url]) => {
    tabs.push({ id: label, label: `Output ${label}`, url, isDownloadable: true });
  });

  if (tabs.length === 0) {
    el.playerFormatTabs.innerHTML = "";
    el.videoPlayer.hidden = true;
    el.videoPlayer.removeAttribute("src");
    el.playerEmptyState.hidden = false;
    el.playerEmptyText.textContent =
      job.status === "awaiting_upload"
        ? "Start processing this video to generate reframed output streams."
        : "Processing video... Reframed output streams will appear here automatically as soon as rendering finishes.";
    el.downloadActiveVideoBtn.hidden = true;
    return;
  }

  // Pick a default tab: prefer rendered output if available, otherwise local preview
  if (!state.activePlayerTab || !tabs.some((t) => t.id === state.activePlayerTab)) {
    const firstOutput = tabs.find((t) => t.id !== "local");
    state.activePlayerTab = (firstOutput || tabs[0]).id;
  }

  el.playerFormatTabs.innerHTML = tabs
    .map(
      (t) => `
      <button
        type="button"
        class="format-tab ${t.id === state.activePlayerTab ? "active" : ""}"
        data-tab-id="${escapeHtml(t.id)}"
      >
        ${escapeHtml(t.label)}
      </button>
    `
    )
    .join("");

  el.playerFormatTabs.querySelectorAll(".format-tab[data-tab-id]").forEach((btn) => {
    btn.addEventListener("click", () => switchPlayerTab(btn.dataset.tabId));
  });

  const activeObj = tabs.find((t) => t.id === state.activePlayerTab) || tabs[0];
  applyVideoSource(activeObj, job);
}

function switchPlayerTab(tabId) {
  state.activePlayerTab = tabId;
  syncVideoPlayerTabs();
}

function applyVideoSource(tabObj, job) {
  el.playerEmptyState.hidden = true;
  el.videoPlayer.hidden = false;

  if (el.videoPlayer.dataset.currentUrl !== tabObj.url) {
    const prevTime = el.videoPlayer.currentTime || 0;
    const wasPlaying = !el.videoPlayer.paused;
    el.videoPlayer.dataset.currentUrl = tabObj.url;
    el.videoPlayer.src = tabObj.url;
    el.videoPlayer.addEventListener(
      "loadedmetadata",
      () => {
        if (prevTime > 0 && prevTime < el.videoPlayer.duration) {
          el.videoPlayer.currentTime = prevTime;
        }
        el.videoPlayer.playbackRate = Number(el.playbackRateSelect.value || 1);
        if (wasPlaying) el.videoPlayer.play().catch(() => {});
      },
      { once: true }
    );
  }

  if (tabObj.isDownloadable) {
    el.downloadActiveVideoBtn.hidden = false;
    el.downloadActiveVideoBtn.href = tabObj.url;
    el.downloadActiveVideoBtn.download = `${(job.filename || "video").replace(/\.[^.]+$/, "")}_${tabObj.id}.mp4`;
    el.downloadBtnLabel.textContent = `Download ${tabObj.id} MP4`;
  } else {
    el.downloadActiveVideoBtn.hidden = true;
  }
}

// ============================================================================
// Layout & Control Event Listeners
// ============================================================================
function togglePlayerDrawer(forceCollapse) {
  state.playerCollapsed =
    typeof forceCollapse === "boolean" ? forceCollapse : !state.playerCollapsed;
  el.videoDrawer.classList.toggle("collapsed", state.playerCollapsed);
  el.togglePlayerBtn.classList.toggle("active", !state.playerCollapsed);
  el.togglePlayerLabel.textContent = state.playerCollapsed ? "Show Player" : "Hide Player";
}

function toggleSidebar(forceCollapse) {
  state.sidebarCollapsed =
    typeof forceCollapse === "boolean" ? forceCollapse : !state.sidebarCollapsed;
  el.sidebar.classList.toggle("collapsed", state.sidebarCollapsed);
  el.expandSidebarBtn.hidden = !state.sidebarCollapsed;
}

function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("fovea_theme", theme);
}

function initEvents() {
  el.collapseSidebarBtn.addEventListener("click", () => toggleSidebar(true));
  el.expandSidebarBtn.addEventListener("click", () => toggleSidebar(false));
  el.togglePlayerBtn.addEventListener("click", () => togglePlayerDrawer());
  el.collapsePlayerBtn.addEventListener("click", () => togglePlayerDrawer(true));

  el.newUploadBtn.addEventListener("click", () => showUploadView());
  el.refreshJobsBtn.addEventListener("click", () => {
    checkHealthAndConfig();
    fetchJobsList();
  });

  el.toggleThemeBtn.addEventListener("click", () => {
    applyTheme(state.theme === "dark" ? "light" : "dark");
  });

  el.searchInput.addEventListener("input", (e) => {
    state.searchQuery = e.target.value || "";
    renderSidebarLists();
  });

  // Upload Dropzone events
  el.dropzoneCard.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) handleFileUpload(file);
    e.target.value = "";
  });

  el.dropzoneCard.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.dropzoneCard.classList.add("dragover");
  });
  el.dropzoneCard.addEventListener("dragleave", () => {
    el.dropzoneCard.classList.remove("dragover");
  });
  el.dropzoneCard.addEventListener("drop", (e) => {
    e.preventDefault();
    el.dropzoneCard.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFileUpload(file);
  });

  // Video Player Controls
  el.videoPlayer.addEventListener("timeupdate", () => {
    el.playerTimeReadout.textContent = `${formatClock(el.videoPlayer.currentTime)} / ${formatClock(
      el.videoPlayer.duration || 0
    )}`;
  });
  el.stepBackBtn.addEventListener("click", () => {
    if (!el.videoPlayer.hidden) {
      el.videoPlayer.pause();
      el.videoPlayer.currentTime = Math.max(0, el.videoPlayer.currentTime - 0.04);
    }
  });
  el.stepFwdBtn.addEventListener("click", () => {
    if (!el.videoPlayer.hidden) {
      el.videoPlayer.pause();
      el.videoPlayer.currentTime = Math.min(
        el.videoPlayer.duration || 0,
        el.videoPlayer.currentTime + 0.04
      );
    }
  });
  el.playbackRateSelect.addEventListener("change", (e) => {
    el.videoPlayer.playbackRate = Number(e.target.value || 1);
  });

  // Settings Modal
  el.openSettingsBtn.addEventListener("click", () => {
    el.userIdInput.value = state.userId;
    el.geminiKeyInput.value = state.geminiKey;
    el.settingsModal.hidden = false;
  });
  el.closeSettingsBtn.addEventListener("click", () => {
    el.settingsModal.hidden = true;
  });
  el.saveSettingsBtn.addEventListener("click", () => {
    const cleanedUser = (el.userIdInput.value || "").trim().replace(/[^a-zA-Z0-9_-]/g, "") || "user-default";
    state.userId = cleanedUser;
    state.geminiKey = (el.geminiKeyInput.value || "").trim();
    localStorage.setItem("fovea_user_id", state.userId);
    localStorage.setItem("fovea_gemini_key", state.geminiKey);
    el.currentUserLabel.textContent = `User: ${state.userId}`;
    el.settingsModal.hidden = true;
    showUploadView();
    fetchJobsList();
  });
}

// ============================================================================
// Bootstrap
// ============================================================================
async function init() {
  applyTheme(state.theme);
  el.currentUserLabel.textContent = `User: ${state.userId}`;
  initEvents();
  await checkHealthAndConfig();
  await fetchJobsList();

  // Poll every 3 seconds (matches frontend/README.md recommendation)
  state.pollTimer = setInterval(() => {
    const hasActive = state.jobs.some((j) =>
      ["queued", "running", "rendered"].includes(j.status)
    );
    if (hasActive || state.selectedJobId) {
      fetchJobsList();
    }
  }, 3000);
}

init();
