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
  allFormats: false, drag: null, poll: null, videoFor: null, dismissed: null, asleep: false,
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
setInterval(() => {              // "the page is open and in front": keeps the server awake
  if (document.visibilityState === "visible" && !S.asleep) api("/api/ping").catch(() => {});
}, 20000);

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
  if (p && p.has_video && S.videoFor !== p.id) {
    S.videoFor = p.id;
    video.src = here("/video");
  }
  if (!p || !p.has_video) {
    S.videoFor = null;
    video.removeAttribute("src");
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

const currentEdit = () => (S.p && S.p.ready ? S.p.edits.find((e) => e.name === S.edit) : null) || null;
const duration = () => S.p.shots[S.p.shots.length - 1].end;
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
  const p = S.p, ready = !!(p && p.ready);
  frame.hidden = !(p && p.has_video);
  $("#empty").hidden = !frame.hidden;
  $("#empty").innerHTML = !p
    ? "<h2>Add a video to begin</h2><p>Fovea watches it once, then shows what each format keeps in the picture.</p>"
    : "<h2>Getting this video ready</h2><p>The picture appears here in a moment.</p>";
  $("#controls").hidden = $("#timeline").hidden = $("#panel").hidden = !ready;
  if (ready) {
    renderFormats();
    renderTimeline();
    renderPanel();
  }
  fit();
}

function renderSidebar() {
  $("#projects").innerHTML = S.projects.map((p) => `
    <div class="project ${p.id === S.pid ? "on" : ""}" data-pid="${esc(p.id)}" title="${esc(p.id)}">
      <span class="dot ${p.busy ? "busy" : p.ready ? "ready" : ""}"></span>
      <span class="name">${esc(p.id)}</span>
      <span class="meta">${p.info ? clock(p.info.duration) : ""}</span>
      <button class="icon remove" data-remove="${esc(p.id)}" title="Remove this project">&times;</button>
    </div>`).join("");
}

function renderTop() {
  const p = S.p;
  if (!p) {
    $("#top").innerHTML = '<span class="title">Fovea</span>';
    return;
  }
  const busy = !!(p.job && p.job.status === "running");
  const meta = p.info ? `${p.info.width}×${p.info.height} · ${clock(p.info.duration)}${p.ready ? ` · ${p.shots.length} shots` : ""}` : "";
  const off = busy ? "disabled" : "";
  const buttons = p.ready
    ? `<button class="btn ghost" data-do="relog" ${off} title="Ask Gemini to watch the video again">Watch again</button>
       <button class="btn primary" data-do="render" ${off} title="Render every format, leaving out unchecked shots">Render</button>`
    : `<button class="btn primary" data-do="log" ${off}>Watch this video</button>`;
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
      <input type="checkbox" data-skip="${s.id}" ${skipped ? "" : "checked"} title="Unchecked shots are left out of renders and edits">
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
  const busy = !!(p.job && p.job.status === "running");
  let html = `<div class="line"><input id="seconds" type="number" min="3" max="600" value="${edit ? edit.target : 15}" style="width:72px">
    seconds <button class="btn primary" data-do="edit" ${busy ? "disabled" : ""}>Make an edit</button></div>
    <p class="hint">Gemini picks what to keep, from the checked shots only. Nothing is rendered until you ask.</p>`;
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
      <button class="btn primary small" data-do="renderEdit" ${busy || !edit.keep.length ? "disabled" : ""}>Render this edit</button></div>`;
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
  if (!p.outputs.length) return '<div class="note">Nothing rendered yet. Press Render, or render an edit from the Edit tab.</div>';
  return p.outputs.map((o) => {
    const url = here(`/outputs/${enc(o.name)}`);
    return `<div class="output"><span class="name" title="${esc(o.name)}">${esc(o.name)}</span>
      <span class="tag">${(o.size / 1e6).toFixed(1)} MB</span>
      <a href="${url}" target="_blank" rel="noopener">Play</a><a href="${url}?download=1">Download</a></div>`;
  }).join("");
}

// The video is fitted into the space it has, keeping its shape; the crop window is drawn over it.
function fit() {
  const p = S.p;
  if (!p || !p.info || frame.hidden) return;
  const stage = $("#stage"), w = stage.clientWidth - 32, h = stage.clientHeight - 32;
  const ratio = p.info.width / p.info.height, width = Math.max(50, Math.min(w, h * ratio));
  frame.style.width = `${width}px`;
  frame.style.height = `${width / ratio}px`;
}
new ResizeObserver(fit).observe($("#stage"));

// =========================================================================== every frame
function tick() {
  requestAnimationFrame(tick);
  const p = S.p;
  if (!p || !p.ready) {
    crop.hidden = $("#dimLeft").hidden = $("#dimRight").hidden = true;
    return;
  }
  const t = video.currentTime || 0;
  if (S.segs && !video.paused && !S.drag && !S.segs.some(([a, b]) => t >= a - 0.02 && t < b)) {
    const next = S.segs.find(([a]) => a > t);         // not in a stretch that is played: jump on
    if (next) {
      video.currentTime = next[0];
    } else {
      video.pause();
      video.currentTime = S.segs[0][0];
    }
    return;
  }
  const shot = shotAt(t);
  if (shot.id !== S.shot) {
    S.shot = shot.id;
    document.querySelectorAll(".now").forEach((el) => el.classList.remove("now"));
    document.querySelectorAll(`[data-shot="${shot.id}"]`).forEach((el) => el.classList.add("now"));
    const card = document.querySelector(`.shot[data-shot="${shot.id}"]`);
    if (card && !video.paused) card.scrollIntoView({ block: "nearest" });
  }
  $("#clock").textContent = `${clock(t)} / ${clock(duration())}   ${shot.id}`;
  const line = $("#timeline");
  $("#playhead").style.left = `${16 + clamp(t / duration(), 0, 1) * (line.clientWidth - 32)}px`;

  const track = S.format ? p.tracks[S.format] : null;
  crop.hidden = $("#dimLeft").hidden = $("#dimRight").hidden = !track;
  if (!track) return;
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
const KINDS = { log: "Watching the video", render: "Rendering", edit: "Making the edit", upload: "Adding the video" };

function showProgress(job) {
  const el = $("#progress");
  const key = job ? `${S.pid}:${job.kind}:${job.error}` : "";
  if (!job || job.status === "done" || job.status === "none" || (job.status === "error" && S.dismissed === key)) {
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
$("#new").onclick = () => $("#file").click();
$("#file").onchange = async () => {
  const file = $("#file").files[0];
  $("#file").value = "";
  if (!file) return;
  showProgress({ kind: "upload", status: "running", lines: [file.name] });
  try {
    const made = await api(`/api/projects?name=${enc(file.name)}`, { method: "POST", body: file });
    S.pid = made.id;
    for (let i = 0; i < 60; i++) {                 // the project appears once the file has been copied
      try {
        await openProject(made.id);
        break;
      } catch (e) {
        await sleep(500);
      }
    }
    await refresh();
  } catch (e) {
    showProgress(null);
    toast(e.message);
  }
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
  if (what === "log") run("/log", {});
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
    toast("Make an edit first, in the Edit tab");
    e.target.value = S.mode;
    return;
  }
  S.mode = e.target.value;
  computeSegs();
};

$("#play").onclick = () => (video.paused ? video.play() : video.pause());
video.onplay = video.onpause = () => { $("#play").textContent = video.paused ? "Play" : "Pause"; };
video.onclick = () => $("#play").click();
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

$("#settingsBtn").onclick = () => {
  $("#key").value = "";
  $("#keyNote").textContent = S.settings.has_key
    ? "A key is saved on this computer." : "No key yet. Fovea needs one to watch a video and to make edits.";
  $("#thinking").innerHTML = (S.settings.thinking_levels || []).map((level) =>
    `<option ${level === S.settings.thinking ? "selected" : ""}>${esc(level)}</option>`).join("");
  $("#settings").returnValue = "";
  $("#settings").showModal();
};
$("#settings").addEventListener("close", async () => {
  if ($("#settings").returnValue !== "save") return;
  const body = { thinking: $("#thinking").value };
  if ($("#key").value.trim()) body.gemini_key = $("#key").value.trim();
  try {
    S.settings = (await post("/api/settings", body)).settings;
  } catch (e) {
    toast(e.message);
  }
});

document.addEventListener("keydown", (e) => {
  if (!S.p || !S.p.ready || $("#settings").open || /INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName)) return;
  if (e.code === "Space") {
    e.preventDefault();
    $("#play").click();
  } else if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
    const shots = S.p.shots, i = shots.indexOf(shotAt(video.currentTime || 0));
    const to = shots[clamp(i + (e.code === "ArrowRight" ? 1 : -1), 0, shots.length - 1)];
    video.currentTime = to.start + 0.001;
  }
});

// =========================================================================== start
if (localStorage.getItem("fovea.collapsed")) $("#app").classList.add("collapsed");
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
refresh().catch((e) => { if (!S.asleep) toast(e.message); });
