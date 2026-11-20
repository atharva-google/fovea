// Fovea's page. Plain JavaScript, no build step. The server (server.py) does the work; this file
// shows a video step by step, draws the crop window over it, and sends what the user changes.
//
// A video goes through three steps, each unlocked by the one before it:
//   video     it is read on this computer (shots, what is said); starts by itself when a video is added
//   framing   Gemini watches it once (only when asked); then every format's crop can be seen and moved
//   edit      Gemini picks what a shorter version keeps (only when asked); it can be changed, then made

const $ = (selector) => document.querySelector(selector);
const enc = encodeURIComponent;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const clock = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

const STEPS = [
  ["video", "Video", ""],
  ["framing", "Framing", "Opens once the video is ready"],
  ["edit", "Shorter version", "Opens once Gemini has watched the video"],
];
const S = {
  projects: [], settings: {},   // from the server
  pid: null, p: null,           // the open video
  home: false,                  // the start screen is shown although there are videos
  step: "video",                // which of the three steps is shown
  format: null,                 // the crop window shown, e.g. "9:16"
  panelTab: "main", panelOff: false, // the right panel: its tab, and whether it is hidden
  shot: null,                   // the shot under the playhead
  edit: null, wantEdit: null,   // the shorter version shown
  skipPlay: true, keptPlay: true, segs: null, // what Play plays: everything, or only some stretches
  allFormats: false, drag: null, poll: null, dismissed: null, asleep: false, updating: false,
  videoFor: null, videoKind: null, // which video is loaded: "source", "proxy" or "failed"
  seeking: false,
};
const video = $("#video"), frame = $("#frame"), crop = $("#crop");

// =========================================================================== talking to the server
async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, { ...options, headers: { "X-Fovea": "1", ...(options.headers || {}) } });
  } catch (e) {
    setAsleep(true);
    throw e;
  }
  setAsleep(false);
  const data = (res.headers.get("content-type") || "").includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error((data && data.error) || res.statusText);
  return data;
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });
const here = (rest = "") => `/api/projects/${enc(S.pid)}${rest}`;

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 6000);
}

// The server stops when the page is closed or idle. While it is gone, show the sleep screen and keep
// knocking; when it answers again, carry on.
function setAsleep(on) {
  if (on === S.asleep || (on && S.updating)) return;   // during an update the server is meant to go away
  S.asleep = on;
  $("#asleep").hidden = !on;
  clearInterval(setAsleep.timer);
  if (on) {
    setAsleep.timer = setInterval(async () => {
      try {
        await fetch("/api/ping");
      } catch (e) {
        return;
      }
      setAsleep(false);
      refresh().catch(() => {});
    }, 3000);
  }
}
// The server stays awake only while someone is using the page: it is in front, and was touched in the
// last half minute (or a video is playing). Otherwise it hears nothing, and goes to sleep after the
// idle time set in Settings.
let touched = Date.now();
["pointermove", "pointerdown", "keydown", "wheel"].forEach((kind) =>
  document.addEventListener(kind, () => { touched = Date.now(); }, { passive: true }));
setInterval(() => {
  const inUse = Date.now() - touched < 30000 || !video.paused;
  if (document.visibilityState === "visible" && inUse && !S.asleep) api("/api/ping").catch(() => {});
}, 15000);

// =========================================================================== loading
async function refresh() {
  const state = await api("/api/state");
  S.projects = state.projects;
  S.settings = state.settings;
  const known = S.projects.map((p) => p.id);
  if (!S.home && !known.includes(S.pid)) {
    const last = localStorage.getItem("fovea.project");
    S.pid = known.includes(last) ? last : null;
  }
  if (S.pid && !S.home) await openProject(S.pid);
  else setProject(null);
}

async function openProject(pid) {
  S.home = false;
  S.pid = pid;
  localStorage.setItem("fovea.project", pid);
  setProject(await api(here()));
}

function goHome() {
  video.pause();
  S.home = true;
  S.pid = null;
  localStorage.removeItem("fovea.project");
  setProject(null);
}

const unlocked = (step) => step === "video" || !!(S.p && (step === "framing" ? S.p.prepared : S.p.ready));

function setProject(p) {
  const fresh = (p ? p.id : null) !== setProject.last;      // another video than the one shown before
  setProject.last = p ? p.id : null;
  S.p = p;
  if (!p || !p.has_video) {
    S.videoFor = S.videoKind = null;
    video.removeAttribute("src");
  } else if (S.videoFor !== p.id) {                  // the file itself: it plays as soon as it is added
    S.videoFor = p.id;
    S.videoKind = "source";
    video.src = here("/video");
  } else if (S.videoKind === "failed" && p.has_proxy) {
    S.videoKind = "proxy";                           // the browser could not play the file: the prepared copy
    video.src = here("/video?proxy=1");
  }
  if (p) {
    if (fresh) {
      S.step = p.ready ? "framing" : "video";
      S.panelTab = "main";
      S.format = null;
    }
    if (!unlocked(S.step)) S.step = unlocked("framing") ? "framing" : "video";
  }
  if (p && p.ready) {
    const shown = p.state.formats.filter((f) => p.tracks[f]);
    if (S.format !== null && !shown.includes(S.format)) S.format = shown[0] || null;
    if (S.format === null && fresh) S.format = shown[0] || null;
    const names = p.edits.map((e) => e.name);
    if (S.wantEdit && names.includes(S.wantEdit)) S.edit = S.wantEdit;
    if (!names.includes(S.edit)) S.edit = names[0] || null;
  }
  computeSegs();
  renderAll();
  watchJob();
}
video.addEventListener("error", () => {
  if (!S.videoFor || S.videoKind !== "source") return;
  if (S.p && S.p.has_proxy) {
    S.videoKind = "proxy";
    video.src = here("/video?proxy=1");
  } else {
    S.videoKind = "failed";
    $("#noplay").hidden = false;
  }
});
video.addEventListener("loadedmetadata", () => {
  $("#noplay").hidden = true;
  fit();
});

const working = () => !!(S.p && S.p.ready && S.step !== "video");   // the crop window, timeline and panel are shown
const currentEdit = () => (S.p && S.p.ready ? S.p.edits.find((e) => e.name === S.edit) : null) || null;
const isBusy = () => !!(S.p && S.p.job && S.p.job.status === "running");
const duration = () => (S.p && S.p.ready ? S.p.shots[S.p.shots.length - 1].end
  : (isFinite(video.duration) && video.duration) || (S.p && S.p.info && S.p.info.duration) || 0);
const shotAt = (t) => S.p.shots.find((s) => t >= s.start && t < s.end) || S.p.shots[S.p.shots.length - 1];
const movedTo = (shot) => {
  const focus = S.p.state.focus;
  const own = (focus[S.format] || {})[shot];
  return own != null ? own : (focus["*"] || {})[shot];
};

function computeSegs() {
  const p = S.p;
  S.segs = null;
  if (!working()) return;
  if (S.step === "edit") {
    const edit = currentEdit();
    if (S.keptPlay && edit && edit.segments.length) S.segs = edit.segments;
  } else if (S.skipPlay && p.state.skipped.length) {
    const out = [];
    for (const s of p.shots) {
      if (p.state.skipped.includes(s.id)) continue;
      const last = out[out.length - 1];
      if (last && Math.abs(last[1] - s.start) < 1e-6) last[1] = s.end;
      else out.push([s.start, s.end]);
    }
    S.segs = out.length ? out : null;
  }
}

async function saveState(change) {
  try {
    setProject(await post(here("/state"), change));
  } catch (e) {
    toast(e.message);
  }
}

function setStep(step) {
  if (!S.p || !unlocked(step) || step === S.step) return;
  S.step = step;
  S.panelTab = "main";
  computeSegs();
  renderAll();
}

// =========================================================================== drawing the page
function renderAll() {
  renderSidebar();
  const p = S.p, hasVideo = !!(p && p.has_video), work = working();
  $("#app").classList.toggle("ready", work && !S.panelOff);
  $("#top").hidden = $("#steps").hidden = !p;
  if (p) {
    const meta = p.info ? `${p.info.width}×${p.info.height} · ${clock(p.info.duration)}` : "";
    $("#top").innerHTML = `<span class="title">${esc(p.name)}</span><span class="meta">${esc(meta)}</span>`;
    $("#steps").innerHTML = STEPS.map(([id, label, locked], i) => `<button data-step="${id}"
      class="${id === S.step ? "on" : ""}" ${unlocked(id) ? "" : `disabled title="${locked}"`}>${i + 1}. ${label}</button>`).join("");
  }
  frame.hidden = !hasVideo;
  $("#empty").hidden = hasVideo;
  if (!hasVideo) $("#empty").innerHTML = welcomeHtml();
  $("#noplay").hidden = S.videoKind !== "failed";
  $("#player").hidden = !hasVideo;
  $("#controls").hidden = $("#timeline").hidden = !work;
  $("#panel").hidden = !(work && !S.panelOff);
  renderNotice();
  if (work) {
    renderControls();
    renderTimeline();
    renderPanel();
  }
  fit();
}

// The start screen: what to do, in order.
function welcomeHtml() {
  const key = S.settings.has_key;
  return `<img class="logo" src="/icon.png" alt="">
    <h1>Welcome to Fovea</h1>
    <p>Turn one ad into every shape: vertical, square and more. Fovea keeps the right part of the
       picture in view, and can cut a shorter version.</p>
    <div class="steps">
      <div class="step"><span class="num ${key ? "done" : ""}">${key ? "&#10003;" : "1"}</span><div style="flex:1">
        <b>${key ? "Your Gemini key is saved" : "Paste your Gemini key"}</b>
        ${key ? "Fovea uses it when you ask Gemini to watch a video. You can change it under Settings."
    : `Fovea uses Google's Gemini to watch your video.
           <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Get a key here</a>, then paste it:
           <div class="keyrow"><input id="firstKey" type="password" autocomplete="off" placeholder="your key">
             <button class="btn" data-do="saveKey">Save</button></div>`}
      </div></div>
      <div class="step"><span class="num">2</span><div>
        <b>Add a video</b>Choose a file, or drop one anywhere on this page. It stays on this computer.</div></div>
    </div>
    <button class="btn primary big" data-do="add">Add a video</button>
    ${S.projects.length ? '<p class="hint" style="margin-top:16px">Or open one of your videos on the left.</p>' : ""}`;
}

const secondsHtml = (value) => `<input id="seconds" type="number" min="3" max="600" value="${value}" title="About how long, in seconds">`;

// Under the video: what is happening in this step, or the one thing to do next.
function renderNotice() {
  const el = $("#notice"), p = S.p, job = p && p.job;
  const running = (kind) => !!job && job.status === "running" && job.kind === kind;
  const failed = (kind) => !!job && job.status === "error" && job.kind === kind;
  const spin = (text) => `<span class="spinner"></span><span>${text}</span>`;
  let html = "", bad = false;
  const oops = (again) => {
    bad = true;
    return `<span>That did not work. <span class="why">${esc(job.error)}</span></span>
      <button class="btn primary" data-do="${again}">Try again</button>`;
  };
  if (!p) {
    html = "";
  } else if (S.step === "video") {
    if (running("prepare")) html = spin("Getting the video ready: finding the shots and writing down what is said. You can play it meanwhile.");
    else if (p.prepared) {
      html = `<span>This video is ready${p.prep ? `: ${p.prep.shots} shots, ${p.prep.words} spoken words` : ""}.</span>`
        + (p.ready ? "" : '<button class="btn primary" data-do="next">Next: Framing</button>');
    } else if (failed("prepare")) html = oops("prepare");
    else html = '<span>This video is not ready yet.</span><button class="btn primary" data-do="prepare">Get it ready</button>';
  } else if (S.step === "framing" && !p.ready) {
    if (running("log")) html = spin("Gemini is watching this video. It takes a few minutes; you can play it meanwhile.");
    else if (!S.settings.has_key) {
      html = '<span>Gemini needs your key before it can watch this video.</span><button class="btn primary" data-do="key">Add your key</button>';
    } else if (failed("log")) html = oops("log");
    else {
      html = '<span>Gemini watches the video once, to find what matters in each shot.</span>'
        + '<button class="btn primary" data-do="log">Watch this video</button>';
    }
  } else if (S.step === "edit" && !p.edits.length) {
    html = running("edit") ? spin("Gemini is choosing what to keep.")
      : `<span>Make a shorter version of about</span>${secondsHtml(15)}<span>seconds</span>
         <button class="btn primary" data-do="edit" ${isBusy() ? "disabled" : ""}>Make a shorter version</button>`;
  }
  el.hidden = !html;
  el.classList.toggle("bad", bad);
  el.innerHTML = html;
}

function renderSidebar() {
  $("#projects").innerHTML = S.projects.map((p) => `
    <div class="project ${p.id === S.pid ? "on" : ""}" data-pid="${esc(p.id)}" title="${esc(p.id)}">
      <span class="dot ${p.busy ? "busy" : p.ready ? "ready" : ""}"></span>
      <span class="name">${esc(p.id)}</span>
      <span class="meta">${p.info ? clock(p.info.duration) : ""}</span>
      <button class="icon remove" data-remove="${esc(p.id)}" title="Remove this video">&times;</button>
    </div>`).join("") || '<div class="none">No videos yet.</div>';
}

// Under the timeline: the formats, and this step's own buttons.
function renderControls() {
  const p = S.p, off = isBusy() ? "disabled" : "", edit = currentEdit();
  const check = (id, on, label, title) => `<label class="check" title="${title}">
    <input id="${id}" type="checkbox" ${on ? "checked" : ""}> ${label}</label>`;
  const formats = p.state.formats.map((f) => (p.tracks[f]
    ? `<button class="chip ${f === S.format ? "on" : ""}" data-format="${esc(f)}">${esc(f)}</button>`
    : `<button class="chip off" title="As wide as the video: nothing to crop">${esc(f)}</button>`)).join("")
    + '<button class="chip" data-format="+" title="Add a format">+</button>';
  let html = `<span class="caption">Format</span><div class="chips">${formats}</div>`;
  if (S.step === "framing") {
    html += check("allFormats", S.allFormats, "move all formats together", "A drag then moves the picture in every format, not only the one shown");
    if (p.state.skipped.length) html += check("skipPlay", S.skipPlay, "skip unticked shots", "Play the video without the shots you unticked");
    html += `<span class="grow"></span>
      <button class="btn ghost" data-do="relog" ${off} title="Ask Gemini to watch the video again">Watch again</button>
      <button class="btn primary" data-do="render" ${off} title="Make a video in every format, leaving out unticked shots">Make the videos</button>`;
  } else if (edit) {
    html += check("keptPlay", S.keptPlay, "play only what is kept", "Play the shorter version, not the whole video");
    html += `<span class="grow"></span>${secondsHtml(edit.target)}<span class="caption">seconds</span>
      <button class="btn ghost" data-do="edit" ${off} title="Ask Gemini to choose again, for this length">Redo</button>
      <button class="btn primary" data-do="renderEdit" ${off || (edit.keep.length ? "" : "disabled")}
        title="Make this shorter version as a video in every format">Make these videos</button>`;
  }
  $("#controls").innerHTML = html;
}

function renderTimeline() {
  const p = S.p, edit = S.step === "edit" ? currentEdit() : null;
  $("#shotRow").innerHTML = p.shots.map((s) => `<div data-shot="${s.id}" style="flex-grow:${s.end - s.start}"
    class="${p.state.skipped.includes(s.id) ? "skipped" : ""} ${s.id === S.shot ? "now" : ""}"
    title="${s.id}: ${esc(s.shows)}">${s.id}</div>`).join("");
  $("#soundRow").innerHTML = p.audio.map((a) => {
    const kept = edit ? (edit.keep.includes(a.id) ? "kept" : "dropped") : "";
    return `<div style="flex-grow:${a.end - a.start}" class="${a.kind === "other" ? "" : "speech"} ${kept}"
      title="${a.id}: ${esc(a.heard)}"></div>`;
  }).join("");
}

function renderPanel() {
  const body = $("#panelBody"), at = body.scrollTop, main = S.step === "edit" ? "What is kept" : "Shots";
  $("#tabs").innerHTML = `<button data-tab="main" class="${S.panelTab === "main" ? "on" : ""}">${main}</button>
    <button data-tab="outputs" class="${S.panelTab === "outputs" ? "on" : ""}">Finished videos</button>`;
  body.innerHTML = S.panelTab === "outputs" ? outputsHtml() : S.step === "edit" ? editHtml() : shotsHtml();
  body.scrollTop = at;
}

function shotsHtml() {
  const p = S.p;
  const names = Object.fromEntries(p.cast.map((c) => [c.id, c.ad_name || c.real_name || c.id]));
  const cast = p.cast.map((c) => `<div class="note"><span class="mono">${esc(c.id)}</span>
    ${esc([c.ad_name, c.real_name].filter(Boolean).join(" / "))} <span class="tag">${esc(c.description)}</span></div>`).join("");
  const shots = p.shots.map((s) => {
    const skipped = p.state.skipped.includes(s.id);
    const said = p.audio.filter((a) => a.kind !== "other" && a.shots[0] === s.id).map((a) => a.heard).join(" ");
    const moved = movedTo(s.id) != null
      ? `<span class="tag good">picture moved</span><button class="btn small" data-reset="${s.id}">reset</button>` : "";
    return `<div class="shot ${skipped ? "skipped" : ""} ${s.id === S.shot ? "now" : ""}" data-shot="${s.id}">
      <input type="checkbox" data-skip="${s.id}" ${skipped ? "" : "checked"} title="Unticked shots are left out of the videos Fovea makes">
      <img loading="lazy" alt="" src="${here(`/thumb/${s.id}.jpg`)}">
      <div class="body">
        <div class="head"><span class="mono">${s.id}</span><span class="tag">${clock(s.start)} · ${(s.end - s.start).toFixed(1)}s</span>
          <span class="tag">${esc(s.cast.map((c) => names[c] || c).join(", "))}</span>${moved}</div>
        <div class="shows">${esc(s.shows)}</div>
        ${said ? `<div class="said">${esc(said)}</div>` : ""}
      </div></div>`;
  }).join("");
  return `<div class="note">${esc(p.summary)}</div><div class="section">Cast</div>${cast}
    <div class="section">Shots</div>${shots}`;
}

function editHtml() {
  const p = S.p, edit = currentEdit();
  if (!edit) return '<div class="note">No shorter version yet.</div>';
  const why = Object.fromEntries(edit.must_keep.map((m) => [m.id, m.why]));
  const gone = new Set(p.state.skipped);
  let html = p.edits.length > 1 ? `<div class="chips" style="padding:4px">${p.edits.map((e) => `<button
    class="chip ${e.name === S.edit ? "on" : ""}" data-edit="${esc(e.name)}">${e.target}s</button>`).join("")}</div>` : "";
  html += `<div class="note">${esc(edit.story)}</div>
    <div class="line"><span class="mono">${edit.seconds.toFixed(1)}s</span><span class="tag">aiming for about ${edit.target}s</span></div>`;
  html += edit.issues.map((i) => `<div class="tag warn" style="padding:2px 4px">${esc(i)}</div>`).join("");
  html += '<div class="section">Tick to keep, untick to drop</div>';
  html += p.audio.map((a) => {
    const kept = edit.keep.includes(a.id), locked = a.shots.some((s) => gone.has(s)) && !kept;
    return `<label class="unit ${kept ? "kept" : ""} ${locked ? "locked" : ""}">
      <input type="checkbox" data-keep="${a.id}" ${kept ? "checked" : ""} ${locked ? "disabled" : ""}>
      <a class="mono" data-seek="${a.start}" title="Go to this moment">${a.id}</a>
      <span class="text">${a.speaker ? `<span class="tag">${esc(a.speaker)}</span> ` : ""}${esc(a.heard)}
        ${why[a.id] ? `<div class="why">${esc(why[a.id])}</div>` : ""}</span></label>`;
  }).join("");
  return html;
}

// The finished videos of this step: the full-length ones under Framing, the shorter ones under Shorter version.
function outputsHtml() {
  const short = S.step === "edit";
  const files = S.p.outputs.filter((o) => o.name.includes("_synced_") === short);
  if (!files.length) {
    return `<div class="note">No finished videos here yet. Press "${short ? "Make these videos" : "Make the videos"}".</div>`;
  }
  return files.map((o) => {
    const url = here(`/outputs/${enc(o.name)}`);
    return `<div class="output"><span class="name" title="${esc(o.name)}">${esc(o.name)}</span>
      <span class="tag">${(o.size / 1e6).toFixed(1)} MB</span>
      <a href="${url}" target="_blank" rel="noopener">Play</a><a href="${url}?download=1">Download</a></div>`;
  }).join("");
}

// The video is fitted into the space it has, keeping its own shape; the crop window is drawn over it.
function fit() {
  const p = S.p;
  if (!p || frame.hidden) return;
  const ratio = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight
    : p.info ? p.info.width / p.info.height : 16 / 9;
  const stage = $("#stage"), w = stage.clientWidth - 32, h = stage.clientHeight - 32;
  const width = Math.max(80, Math.floor(Math.min(w, h * ratio)));
  frame.style.width = `${width}px`;
  frame.style.height = `${Math.round(width / ratio)}px`;
}
new ResizeObserver(fit).observe($("#stage"));

// =========================================================================== every frame
function tick() {
  requestAnimationFrame(tick);
  const p = S.p, hide = (on) => { crop.hidden = $("#dimLeft").hidden = $("#dimRight").hidden = on; };
  if (!p || !p.has_video) return hide(true);
  const t = video.currentTime || 0, total = duration();
  if (!S.seeking && total) $("#seek").value = Math.round(clamp(t / total, 0, 1) * 1000);
  if (!working()) {
    $("#clock").textContent = `${clock(t)} / ${clock(total)}`;
    return hide(true);
  }
  if (S.segs && !video.paused && !S.drag && !S.segs.some(([a, b]) => t >= a - 0.02 && t < b)) {
    const next = S.segs.find(([a]) => a > t);         // not in a stretch that is played: jump on
    if (next) {
      video.currentTime = next[0];
    } else {
      video.pause();
      video.currentTime = S.segs[0][0];
    }
    return undefined;
  }
  const shot = shotAt(t);
  if (shot.id !== S.shot) {
    S.shot = shot.id;
    document.querySelectorAll(".now").forEach((el) => el.classList.remove("now"));
    document.querySelectorAll(`[data-shot="${shot.id}"]`).forEach((el) => el.classList.add("now"));
    const card = document.querySelector(`.shot[data-shot="${shot.id}"]`);
    if (card && !video.paused) card.scrollIntoView({ block: "nearest" });
  }
  $("#clock").textContent = `${clock(t)} / ${clock(total)}   ${shot.id}`;
  const line = $("#timeline");
  $("#playhead").style.left = `${16 + clamp(t / total, 0, 1) * (line.clientWidth - 32)}px`;

  const track = S.format ? p.tracks[S.format] : null;
  hide(!track);
  if (!track) return undefined;
  let left;
  if (S.drag) {
    left = S.drag.left;
  } else {
    const [a, b, x0, x1] = track.pieces.find((piece) => t >= piece[0] && t < piece[1]) || track.pieces[track.pieces.length - 1];
    left = x0 + (x1 - x0) * clamp((t - a) / (b - a), 0, 1);
  }
  crop.style.left = `${left * 100}%`;
  crop.style.width = `${track.w * 100}%`;
  $("#dimLeft").style.width = `${left * 100}%`;
  $("#dimRight").style.width = `${Math.max(0, 1 - left - track.w) * 100}%`;
  crop.classList.toggle("moved", movedTo(shot.id) != null);
  return undefined;
}
requestAnimationFrame(tick);

// =========================================================================== dragging the crop window
// A drag holds the picture where it is dropped for the whole shot: in this format, or in all of them.
crop.addEventListener("pointerdown", (e) => {
  const track = working() && S.format ? S.p.tracks[S.format] : null;
  if (!track || !S.shot) return;
  video.pause();
  crop.setPointerCapture(e.pointerId);
  const left = parseFloat(crop.style.left) / 100 || 0;
  S.drag = { x: e.clientX, from: left, left, w: track.w, shot: S.shot, far: false, saving: false };
});
crop.addEventListener("pointermove", (e) => {
  const d = S.drag;
  if (!d || d.saving) return;
  const dx = (e.clientX - d.x) / frame.clientWidth;
  d.far = d.far || Math.abs(dx) > 0.004;
  d.left = clamp(d.from + dx, 0, 1 - d.w);
});
crop.addEventListener("pointerup", async () => {
  const d = S.drag;
  if (!d || d.saving) return;
  if (!d.far) {
    S.drag = null;
    return;
  }
  d.saving = true;
  const focus = JSON.parse(JSON.stringify(S.p.state.focus));
  const key = S.allFormats ? "*" : S.format;
  if (S.allFormats) Object.keys(focus).forEach((f) => { if (f !== "*") delete focus[f][d.shot]; });
  focus[key] = { ...(focus[key] || {}), [d.shot]: Math.round((d.left + d.w / 2) * 100) };
  await saveState({ focus });
  S.drag = null;
});
crop.addEventListener("pointercancel", () => { S.drag = null; });

// =========================================================================== jobs
const KINDS = {
  update: "Updating Fovea", upload: "Adding the video", prepare: "Getting the video ready",
  log: "Gemini is watching the video", render: "Making the videos", edit: "Making the shorter version",
};

function showProgress(job) {
  const el = $("#progress");
  const key = job ? `${S.pid}:${job.kind}:${job.error}` : "";
  // a failed first or second step is explained in the notice under the video, with its "Try again"
  const said = job && job.status === "error" && (job.kind === "prepare" || (job.kind === "log" && S.p && !S.p.ready));
  if (!job || job.status === "done" || job.status === "none" || said || (job.status === "error" && S.dismissed === key)) {
    el.hidden = true;
    return;
  }
  const failed = job.status === "error";
  el.hidden = false;
  el.classList.toggle("failed", failed);
  el.innerHTML = failed
    ? `<div class="what">${esc(KINDS[job.kind] || job.kind)}: that did not work<span class="grow"></span>
       <button class="btn small" data-dismiss="${esc(key)}">Close</button></div><div class="lines">${esc(job.error)}</div>`
    : `<div class="what"><span class="spinner"></span>${esc(KINDS[job.kind] || job.kind)}…</div>
       <div class="lines">${esc(job.lines.slice(-6).join("\n"))}</div>`;
}

function watchJob() {
  const job = S.p ? S.p.job : null;
  showProgress(job);
  clearInterval(S.poll);
  S.poll = null;
  if (!job || job.status !== "running") return;
  const pid = S.pid;
  S.poll = setInterval(async () => {
    let now;
    try {
      now = await api(`/api/projects/${enc(pid)}/job`);
    } catch (e) {
      return;
    }
    if (pid !== S.pid) return;
    S.p.job = now;
    showProgress(now);
    if (now.status === "running") return;
    clearInterval(S.poll);
    S.poll = null;
    if (now.status === "done" && now.kind === "render") S.panelTab = "outputs";
    await refresh().catch((e) => toast(e.message));
  }, 1500);
}

async function run(path, body) {
  try {
    setProject(await post(here(path), body));
  } catch (e) {
    toast(e.message);
  }
}

// =========================================================================== buttons
// Every button that does something carries data-do; this is the one place where they are acted on.
// Gemini is only ever asked from here ("log", "relog", "edit").
function act(what) {
  if (what === "add") return $("#file").click();
  if (what === "key") return openSettings();
  if (what === "saveKey") {
    const key = $("#firstKey").value.trim();
    return key ? saveSettings({ gemini_key: key }) : toast("Paste your key first");
  }
  if (what === "prepare") return run("/prepare", {});
  if (what === "next") return setStep("framing");
  if (what === "log") return run("/log", {});
  if (what === "relog") {
    return confirm("Ask Gemini to watch the video again? This replaces what it found and takes a few minutes.")
      && run("/log", { refresh: true });
  }
  if (what === "render") return run("/render", {});
  if (what === "renderEdit") return run("/render", { edit: S.edit });
  if (what === "edit") {
    const seconds = parseFloat($("#seconds").value);
    if (!(seconds >= 1)) return toast("Give a length in seconds");
    if (!S.settings.has_key) return openSettings();
    S.wantEdit = `synced_${seconds}s`;
    return run("/edit", { seconds });
  }
  return undefined;
}
document.addEventListener("click", (e) => {
  const button = e.target.closest ? e.target.closest("[data-do]") : null;
  if (button && !button.disabled) act(button.dataset.do);
});

function toggleSidebar() {
  const on = $("#app").classList.toggle("collapsed");
  localStorage.setItem("fovea.collapsed", on ? "1" : "");
  fit();
}
function togglePanel() {
  S.panelOff = !S.panelOff;
  localStorage.setItem("fovea.panelOff", S.panelOff ? "1" : "");
  renderAll();
}
$("#collapse").onclick = toggleSidebar;
$("#home").onclick = goHome;
$("#steps").onclick = (e) => { if (e.target.dataset.step) setStep(e.target.dataset.step); };

// Adding a video: from the button, the start screen, or by dropping a file on the page.
async function addVideo(file) {
  if (!file) return;
  const name = file.name.replace(/\.[^.]*$/, "");
  if (S.projects.some((p) => p.id === name)
      && !confirm(`You already have a video called "${name}". If this is a different file, it replaces that one. Continue?`)) return;
  showProgress({ kind: "upload", status: "running", lines: [file.name] });
  try {
    const made = await api(`/api/projects?name=${enc(file.name)}`, { method: "POST", body: file });
    S.home = false;
    S.pid = made.id;
    S.videoFor = null;
    setProject.last = null;
    await refresh();
  } catch (e) {
    toast(e.message);
  }
  if (!isBusy()) showProgress(null);
}
$("#new").onclick = () => $("#file").click();
$("#file").onchange = () => {
  const file = $("#file").files[0];
  $("#file").value = "";
  addVideo(file);
};
let dragDepth = 0;
document.addEventListener("dragenter", (e) => {
  if (![...(e.dataTransfer ? e.dataTransfer.types : [])].includes("Files")) return;
  dragDepth += 1;
  $("#drop").hidden = false;
});
document.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("#drop").hidden = true;
});
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("#drop").hidden = true;
  addVideo(e.dataTransfer && e.dataTransfer.files[0]);
});

$("#projects").onclick = async (e) => {
  const remove = e.target.closest("[data-remove]"), row = e.target.closest("[data-pid]");
  try {
    if (remove) {
      const pid = remove.dataset.remove;
      if (!confirm(`Remove "${pid}" and everything made for it? Your original file is not touched.`)) return;
      await api(`/api/projects/${enc(pid)}`, { method: "DELETE" });
      if (pid === S.pid) S.pid = null;
      await refresh();
    } else if (row && row.dataset.pid !== S.pid) {
      video.pause();
      await openProject(row.dataset.pid);
    }
  } catch (err) {
    toast(err.message);
  }
};

$("#controls").onclick = (e) => {
  const chip = e.target.closest("[data-format]");
  if (!chip) return;
  if (chip.dataset.format === "+") {
    const added = prompt("Add a format, as WIDTH:HEIGHT (for example 4:5)");
    if (added) {
      S.format = added.trim().toLowerCase().replace("x", ":");
      saveState({ formats: [...S.p.state.formats, added.trim()] });
    }
    return;
  }
  S.format = S.format === chip.dataset.format ? null : chip.dataset.format;   // a second click hides the window
  renderControls();
  if (S.panelTab === "main") renderPanel();
};
$("#controls").onchange = (e) => {
  const id = e.target.id;
  if (id !== "allFormats" && id !== "skipPlay" && id !== "keptPlay") return;
  S[id] = e.target.checked;
  computeSegs();
};

$("#play").onclick = () => (video.paused ? video.play() : video.pause());
video.onplay = video.onpause = () => { $("#play").textContent = video.paused ? "Play" : "Pause"; };
video.onclick = () => $("#play").click();
$("#mute").onclick = () => {
  video.muted = !video.muted;
  $("#mute").textContent = video.muted ? "Sound off" : "Sound on";
};
$("#seek").addEventListener("pointerdown", () => { S.seeking = true; });
$("#seek").addEventListener("input", () => { video.currentTime = ($("#seek").value / 1000) * duration(); });
$("#seek").addEventListener("change", () => { S.seeking = false; });
$("#timeline").onclick = (e) => {
  const box = $("#shotRow").getBoundingClientRect();
  video.currentTime = clamp((e.clientX - box.left) / box.width, 0, 1) * duration();
};

$("#tabs").onclick = (e) => {
  if (!e.target.dataset.tab) return;
  S.panelTab = e.target.dataset.tab;
  renderPanel();
};

$("#panelBody").onclick = (e) => {
  const p = S.p, target = e.target;
  if (target.closest("[data-skip], [data-keep]")) return;          // handled when the box changes
  const reset = target.closest("[data-reset]"), seek = target.closest("[data-seek]"), pick = target.closest("[data-edit]");
  const card = target.closest(".shot");
  if (reset) {
    const focus = JSON.parse(JSON.stringify(p.state.focus));
    [S.format, "*"].forEach((f) => { if (focus[f]) delete focus[f][reset.dataset.reset]; });
    saveState({ focus });
  } else if (seek) {
    e.preventDefault();
    video.currentTime = parseFloat(seek.dataset.seek);
  } else if (pick) {
    S.edit = S.wantEdit = pick.dataset.edit;
    computeSegs();
    renderAll();
  } else if (card) {
    video.currentTime = p.shots.find((s) => s.id === card.dataset.shot).start + 0.001;
  }
};

$("#panelBody").onchange = async (e) => {
  const p = S.p, box = e.target;
  if (box.dataset.skip) {
    const skipped = new Set(p.state.skipped);
    if (box.checked) skipped.delete(box.dataset.skip);
    else skipped.add(box.dataset.skip);
    await saveState({ skipped: [...skipped] });
  } else if (box.dataset.keep) {
    const keep = new Set(currentEdit().keep);
    if (box.checked) keep.add(box.dataset.keep);
    else keep.delete(box.dataset.keep);
    try {
      setProject(await post(here(`/edits/${enc(S.edit)}`), { keep: [...keep] }));
    } catch (err) {
      toast(err.message);
    }
  }
};

$("#progress").onclick = (e) => {
  const close = e.target.closest("[data-dismiss]");
  if (!close) return;
  S.dismissed = close.dataset.dismiss;
  $("#progress").hidden = true;
};

// =========================================================================== settings
async function saveSettings(body) {
  try {
    S.settings = (await post("/api/settings", body)).settings;
  } catch (e) {
    return toast(e.message);
  }
  return renderAll();
}
function showPane(pane) {
  $("#paneGeneral").hidden = pane !== "general";
  $("#paneShortcuts").hidden = pane !== "shortcuts";
  $("#settingsNav").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.pane === pane));
}
function openSettings() {
  $("#key").value = "";
  $("#keyNote").innerHTML = (S.settings.has_key ? "A key is saved. " : "No key yet. ")
    + 'It is kept in a file on this computer and only ever sent to Google\'s Gemini. '
    + '<a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Get a key</a>';
  $("#removeKey").hidden = !S.settings.has_key;
  $("#thinking").innerHTML = (S.settings.thinking_levels || []).map((level) =>
    `<option ${level === S.settings.thinking ? "selected" : ""}>${esc(level)}</option>`).join("");
  $("#idle").value = Math.round(S.settings.idle_minutes || 10);
  $("#version").textContent = S.settings.version ? `version ${S.settings.version}` : "";
  showPane("general");
  $("#settings").returnValue = "";
  $("#settings").showModal();
}
$("#settingsBtn").onclick = openSettings;
$("#settingsNav").onclick = (e) => { if (e.target.dataset.pane) showPane(e.target.dataset.pane); };
$("#cancelSettings").onclick = () => $("#settings").close("cancel");
$("#updateBtn").onclick = () => $("#settings").close("update");
$("#removeKey").onclick = async () => {
  if (!confirm("Remove the saved Gemini key from this computer?")) return;
  $("#settings").close("cancel");
  await saveSettings({ gemini_key: "" });
  if (S.settings.has_key) toast("The saved key is removed, but this computer also has one set as GEMINI_API_KEY");
};

// Update: the server fetches the latest Fovea and starts again; the page waits for it and reloads.
async function updateFovea() {
  const show = (state) => showProgress({ kind: "update", status: "running", lines: state.lines || [] });
  const ask = async (method) => {
    const res = await fetch("/api/update", { method, headers: { "X-Fovea": "1" } });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  };
  let state;
  try {
    state = await ask("POST");
  } catch (e) {
    return toast(e.message);
  }
  S.updating = true;                       // the server going away is expected: no sleep screen
  for (let gone = 0; gone < 120;) {
    show(state);
    if (state.status === "current" || state.status === "error") break;
    await sleep(1000);
    try {
      const now = await ask("GET");
      if (state.status === "restarting" && now.status === "idle") return location.reload();   // the new one answered
      state = now;
    } catch (e) {
      state = { status: "restarting", lines: [...(state.lines || []), "Starting the new version ..."].slice(-6) };
      gone += 1;
    }
  }
  S.updating = false;
  showProgress(null);
  if (state.status === "current") return toast("Fovea is already up to date");
  return toast(state.status === "error" ? `The update did not work: ${state.error}`
    : "Fovea did not come back. Click the Fovea icon to start it.");
}

$("#settings").addEventListener("close", () => {
  if ($("#settings").returnValue === "update") return updateFovea();
  if ($("#settings").returnValue !== "save") return undefined;
  const body = { thinking: $("#thinking").value };
  if (parseFloat($("#idle").value) >= 1) body.idle_minutes = parseFloat($("#idle").value);
  if ($("#key").value.trim()) body.gemini_key = $("#key").value.trim();
  return saveSettings(body);
});

// =========================================================================== keyboard (listed under Settings > Shortcuts)
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.code === "KeyB") {       // Ctrl+B: the list of videos; with Shift: the details
    e.preventDefault();
    return e.shiftKey ? togglePanel() : toggleSidebar();
  }
  if (!S.p || !S.p.has_video || $("#settings").open || /INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName)) return undefined;
  if (e.code === "Space") {
    e.preventDefault();
    $("#play").click();
  } else if (working() && (e.code === "ArrowRight" || e.code === "ArrowLeft")) {
    const shots = S.p.shots, i = shots.indexOf(shotAt(video.currentTime || 0));
    const to = shots[clamp(i + (e.code === "ArrowRight" ? 1 : -1), 0, shots.length - 1)];
    video.currentTime = to.start + 0.001;
  }
  return undefined;
});

// =========================================================================== start
if (localStorage.getItem("fovea.collapsed")) $("#app").classList.add("collapsed");
S.panelOff = !!localStorage.getItem("fovea.panelOff");
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
refresh().catch((e) => { if (!S.asleep) toast(e.message); });
