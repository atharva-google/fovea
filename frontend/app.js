// Fovea's page. Plain JavaScript, no build step. The server (server.py) does the work; this file
// shows a project, draws the crop window over the video, and sends what the user changes.

const $ = (selector) => document.querySelector(selector);
const enc = encodeURIComponent;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const clock = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

const S = {
  projects: [], settings: {}, // from the server
  pid: null, p: null,         // the open project
  format: null,               // the crop window shown, e.g. "9:16"
  tab: "shots", shot: null,   // right panel; the shot under the playhead
  edit: null, wantEdit: null, // the edit shown in the Edit tab
  mode: "original", segs: null, // what Play plays: everything, or only some stretches
  allFormats: false, drag: null, poll: null, dismissed: null, asleep: false,
  videoFor: null, videoKind: null, // which project's video is loaded: "source", "proxy" or "failed"
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
  if (on === S.asleep) return;
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
  if (!known.includes(S.pid)) S.pid = known.includes(localStorage.getItem("fovea.project"))
    ? localStorage.getItem("fovea.project") : known[0] || null;
  if (S.pid) await openProject(S.pid);
  else setProject(null);
}

async function openProject(pid) {
  S.pid = pid;
  localStorage.setItem("fovea.project", pid);
  setProject(await api(here()));
}

function setProject(p) {
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
  if (p && p.ready) {
    const shown = p.state.formats.filter((f) => p.tracks[f]);
    if (S.format !== null && !shown.includes(S.format)) S.format = shown[0] || null;
    if (S.format === null && S.pid !== setProject.last) S.format = shown[0] || null;
    const names = p.edits.map((e) => e.name);
    if (S.wantEdit && names.includes(S.wantEdit)) S.edit = S.wantEdit;
    if (!names.includes(S.edit)) S.edit = names[0] || null;
  }
  setProject.last = S.pid;
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
  if (!p || !p.ready) return;
  if (S.mode === "edit") {
    const edit = currentEdit();
    S.segs = edit && edit.segments.length ? edit.segments : null;
  } else if (S.mode === "selected") {
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

// =========================================================================== drawing the page
function renderAll() {
  renderSidebar();
  renderTop();
  const p = S.p, hasVideo = !!(p && p.has_video), ready = !!(p && p.ready);
  $("#app").classList.toggle("ready", ready);
  frame.hidden = !hasVideo;
  $("#empty").hidden = hasVideo;
  if (!hasVideo) $("#empty").innerHTML = welcomeHtml();
  $("#noplay").hidden = S.videoKind !== "failed";
  $("#player").hidden = !hasVideo;
  $("#controls").hidden = $("#timeline").hidden = $("#panel").hidden = !ready;
  renderNotice();
  if (ready) {
    renderFormats();
    renderTimeline();
    renderPanel();
  }
  fit();
}

// The first screen: what to do, in order. Shown while there is no video.
function welcomeHtml() {
  const key = S.settings.has_key;
  return `<img class="logo" src="/icon.png" alt="">
    <h1>Welcome to Fovea</h1>
    <p>Turn one ad into every shape: vertical, square and more. Fovea watches your video once and keeps
       the right part of the picture in view.</p>
    <div class="steps">
      <div class="step"><span class="num ${key ? "done" : ""}">${key ? "&#10003;" : "1"}</span><div style="flex:1">
        <b>${key ? "Your Gemini key is saved" : "Paste your Gemini key"}</b>
        ${key ? "Fovea uses it to watch your videos. You can change it under Settings."
    : `Fovea uses Google's Gemini to watch your video.
           <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Get a key here</a>, then paste it:
           <div class="keyrow"><input id="firstKey" type="password" autocomplete="off" placeholder="your key">
             <button class="btn" data-do="saveKey">Save</button></div>`}
      </div></div>
      <div class="step"><span class="num">2</span><div>
        <b>Add a video</b>Choose a file, or drop one anywhere on this page. It stays on this computer.</div></div>
    </div>
    <button class="btn primary big" data-do="add">Add a video</button>`;
}

// Under the video, while it has not been watched yet: what is happening, or the one thing to do next.
function renderNotice() {
  const el = $("#notice"), p = S.p;
  if (!p || p.ready) {
    el.hidden = true;
    return;
  }
  const failed = !!(p.job && p.job.status === "error");
  el.hidden = false;
  el.classList.toggle("bad", failed && S.settings.has_key);
  if (isBusy()) {
    el.innerHTML = '<span class="spinner"></span><span>Fovea is watching this video. It takes a few minutes; you can play it meanwhile.</span>';
  } else if (!S.settings.has_key) {
    el.innerHTML = '<span>Fovea needs your Gemini key before it can watch this video.</span>'
      + '<button class="btn primary" data-do="key">Add your key</button>';
  } else if (failed) {
    el.innerHTML = `<span>That did not work. <span class="why">${esc(p.job.error)}</span></span>
      <button class="btn primary" data-do="log">Try again</button>`;
  } else {
    el.innerHTML = '<span>Fovea has not watched this video yet.</span><button class="btn primary" data-do="log">Watch this video</button>';
  }
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

function renderTop() {
  const p = S.p;
  if (!p) {
    $("#top").innerHTML = '<span class="title">Fovea</span>';
    return;
  }
  const meta = p.info ? `${p.info.width}×${p.info.height} · ${clock(p.info.duration)}${p.ready ? ` · ${p.shots.length} shots` : ""}` : "";
  const off = isBusy() ? "disabled" : "";
  const buttons = !p.ready ? ""
    : `<button class="btn ghost" data-do="relog" ${off} title="Ask Gemini to watch the video again">Watch again</button>
       <button class="btn primary" data-do="render" ${off} title="Make a video in every format, leaving out unticked shots">Make the videos</button>`;
  $("#top").innerHTML = `<span class="title">${esc(p.name)}</span><span class="meta">${esc(meta)}</span>
    <span class="grow"></span>${buttons}`;
}

function renderFormats() {
  const p = S.p;
  $("#formats").innerHTML = p.state.formats.map((f) => (p.tracks[f]
    ? `<button class="chip ${f === S.format ? "on" : ""}" data-format="${esc(f)}">${esc(f)}</button>`
    : `<button class="chip off" title="As wide as the video: nothing to crop">${esc(f)}</button>`)).join("")
    + '<button class="chip" data-format="+" title="Add a format">+</button>';
}

function renderTimeline() {
  const p = S.p, edit = S.tab === "edit" ? currentEdit() : null;
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
  const body = $("#panelBody"), at = body.scrollTop;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === S.tab));
  body.innerHTML = S.tab === "shots" ? shotsHtml() : S.tab === "edit" ? editHtml() : outputsHtml();
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
  const p = S.p, edit = currentEdit(), busy = isBusy();
  let html = `<div class="line"><input id="seconds" type="number" min="3" max="600" value="${edit ? edit.target : 15}" style="width:72px">
    seconds <button class="btn primary" data-do="edit" ${busy ? "disabled" : ""}>Make a shorter version</button></div>
    <p class="hint">Gemini picks what to keep, from the ticked shots only. No video is made until you ask.</p>`;
  if (p.edits.length) {
    html += `<div class="chips">${p.edits.map((e) => `<button class="chip ${e.name === S.edit ? "on" : ""}"
      data-edit="${esc(e.name)}">${e.target}s</button>`).join("")}</div>`;
  }
  if (!edit) return html;
  const why = Object.fromEntries(edit.must_keep.map((m) => [m.id, m.why]));
  const gone = new Set(p.state.skipped);
  html += `<div class="note">${esc(edit.story)}</div>
    <div class="line"><span class="mono">${edit.seconds.toFixed(1)}s</span><span class="tag">aiming for about ${edit.target}s</span>
      <button class="btn small" data-do="preview">Preview</button>
      <button class="btn primary small" data-do="renderEdit" ${busy || !edit.keep.length ? "disabled" : ""}>Make these videos</button></div>`;
  html += edit.issues.map((i) => `<div class="tag warn" style="padding:2px 4px">${esc(i)}</div>`).join("");
  html += '<div class="section">What is kept (tick to change)</div>';
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

function outputsHtml() {
  const p = S.p;
  if (!p.outputs.length) return '<div class="note">No finished videos yet. Press "Make the videos" at the top, or make a shorter version first.</div>';
  return p.outputs.map((o) => {
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
  if (!p.ready) {
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
  const track = S.p && S.p.ready && S.format ? S.p.tracks[S.format] : null;
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

// =========================================================================== jobs (log, render, edit)
const KINDS = { log: "Watching the video", render: "Making the videos", edit: "Making the shorter version", upload: "Adding the video" };

function showProgress(job) {
  const el = $("#progress");
  const key = job ? `${S.pid}:${job.kind}:${job.error}` : "";
  const said = job && job.status === "error" && job.kind === "log" && S.p && !S.p.ready;   // the notice under the video says it
  if (!job || job.status === "done" || job.status === "none" || said || (job.status === "error" && S.dismissed === key)) {
    el.hidden = true;
    return;
  }
  const failed = job.status === "error";
  el.hidden = false;
  el.classList.toggle("failed", failed);
  el.innerHTML = failed
    ? `<div class="what">${esc(KINDS[job.kind] || job.kind)} did not work<span style="flex:1"></span>
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
    if (now.status === "done" && now.kind === "render") S.tab = "outputs";
    await refresh().catch((e) => toast(e.message));
  }, 1500);
}

async function run(path, body) {
  try {
    setProject(await post(here(path), body));
    renderSidebar();
  } catch (e) {
    toast(e.message);
  }
}

// =========================================================================== what the user does
$("#collapse").onclick = () => {
  const on = $("#app").classList.toggle("collapsed");
  localStorage.setItem("fovea.collapsed", on ? "1" : "");
  fit();
};
// Adding a video: from the button, the first screen, or by dropping a file on the page.
async function addVideo(file) {
  if (!file) return;
  const name = file.name.replace(/\.[^.]*$/, "");
  if (S.projects.some((p) => p.id === name)
      && !confirm(`You already have a video called "${name}". If this is a different file, it replaces that one. Continue?`)) return;
  showProgress({ kind: "upload", status: "running", lines: [file.name] });
  try {
    const made = await api(`/api/projects?name=${enc(file.name)}`, { method: "POST", body: file });
    S.pid = made.id;
    S.videoFor = null;
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

async function saveSettings(body) {
  try {
    S.settings = (await post("/api/settings", body)).settings;
  } catch (e) {
    return toast(e.message);
  }
  if (S.settings.has_key && S.p && !S.p.ready && !isBusy()) return run("/log", {});   // it was waiting for the key
  return renderAll();
}
function openSettings() {
  $("#key").value = "";
  $("#keyNote").innerHTML = (S.settings.has_key ? "A key is saved. " : "No key yet. ")
    + 'It is kept in a file on this computer and only ever sent to Google\'s Gemini. '
    + '<a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Get a key</a>';
  $("#thinking").innerHTML = (S.settings.thinking_levels || []).map((level) =>
    `<option ${level === S.settings.thinking ? "selected" : ""}>${esc(level)}</option>`).join("");
  $("#idle").value = Math.round(S.settings.idle_minutes || 10);
  $("#settings").returnValue = "";
  $("#settings").showModal();
}
$("#empty").onclick = (e) => {
  const what = (e.target.closest("[data-do]") || { dataset: {} }).dataset.do;
  if (what === "add") $("#file").click();
  if (what === "saveKey") {
    const key = $("#firstKey").value.trim();
    if (key) saveSettings({ gemini_key: key });
    else toast("Paste your key first");
  }
};
$("#notice").onclick = (e) => {
  const what = (e.target.closest("[data-do]") || { dataset: {} }).dataset.do;
  if (what === "key") openSettings();
  if (what === "log") run("/log", {});
};

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

$("#top").onclick = (e) => {
  const what = (e.target.closest("[data-do]") || { dataset: {} }).dataset.do;
  if (what === "render") run("/render", {});
  if (what === "relog" && confirm("Ask Gemini to watch the video again? This replaces what it found and takes a few minutes.")) {
    run("/log", { refresh: true });
  }
};

$("#formats").onclick = (e) => {
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
  renderFormats();
  if (S.tab === "shots") renderPanel();
};
$("#allFormats").onchange = (e) => { S.allFormats = e.target.checked; };
$("#mode").onchange = (e) => {
  if (e.target.value === "edit" && !currentEdit()) {
    toast("Make a shorter version first, in the \"Shorter version\" tab");
    e.target.value = S.mode;
    return;
  }
  S.mode = e.target.value;
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
  const tab = e.target.dataset.tab;
  if (!tab) return;
  S.tab = tab;
  renderPanel();
  renderTimeline();
};

$("#panelBody").onclick = (e) => {
  const p = S.p, target = e.target;
  if (target.closest("[data-skip], [data-keep]")) return;          // handled when the box changes
  const reset = target.closest("[data-reset]"), seek = target.closest("[data-seek]"), pick = target.closest("[data-edit]");
  const act = (target.closest("[data-do]") || { dataset: {} }).dataset.do, card = target.closest(".shot");
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
    renderPanel();
    renderTimeline();
  } else if (act === "edit") {
    const seconds = parseFloat($("#seconds").value);
    if (!(seconds >= 1)) return toast("Give a length in seconds");
    S.wantEdit = `synced_${seconds}s`;
    run("/edit", { seconds });
  } else if (act === "renderEdit") {
    run("/render", { edit: S.edit });
  } else if (act === "preview") {
    S.mode = $("#mode").value = "edit";
    computeSegs();
    if (S.segs) {
      video.currentTime = S.segs[0][0];
      video.play();
    }
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

$("#settingsBtn").onclick = openSettings;
$("#settings").addEventListener("close", () => {
  if ($("#settings").returnValue !== "save") return;
  const body = { thinking: $("#thinking").value };
  if (parseFloat($("#idle").value) >= 1) body.idle_minutes = parseFloat($("#idle").value);
  if ($("#key").value.trim()) body.gemini_key = $("#key").value.trim();
  saveSettings(body);
});

document.addEventListener("keydown", (e) => {
  if (!S.p || !S.p.has_video || $("#settings").open || /INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName)) return;
  if (e.code === "Space") {
    e.preventDefault();
    $("#play").click();
  } else if (S.p.ready && (e.code === "ArrowRight" || e.code === "ArrowLeft")) {
    const shots = S.p.shots, i = shots.indexOf(shotAt(video.currentTime || 0));
    const to = shots[clamp(i + (e.code === "ArrowRight" ? 1 : -1), 0, shots.length - 1)];
    video.currentTime = to.start + 0.001;
  }
});

// =========================================================================== start
if (localStorage.getItem("fovea.collapsed")) $("#app").classList.add("collapsed");
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
refresh().catch((e) => { if (!S.asleep) toast(e.message); });
