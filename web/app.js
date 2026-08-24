"use strict";

const $ = (selector) => document.querySelector(selector);

const state = {
  ctx: null,
  words: [],
  chunks: [],
  sources: [],
  sampleRate: 24000,
  scheduleAt: 0,
  playStart: 0,
  playing: false,
  raf: 0,
};

/* ---------- helpers ---------- */

function setStatus(message, isError) {
  const el = $("#status");
  el.textContent = message;
  el.classList.toggle("error", Boolean(isError));
}

async function api(path, options) {
  const settings = Object.assign({}, options);
  const headers = Object.assign(
    { "Content-Type": "application/json" },
    settings.headers || {}
  );
  const key = $("#apiKey").value.trim();
  if (key) headers.Authorization = "Bearer " + key;
  settings.headers = headers;
  return fetch(path, settings);
}

async function errorMessage(resp) {
  try {
    const body = await resp.json();
    if (body && body.error && body.error.message) return body.error.message;
  } catch (err) {
    /* fall through to the status line below */
  }
  return "Request failed with status " + resp.status;
}

function decodePcm(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function wavBlob(chunks, sampleRate) {
  let length = 0;
  chunks.forEach((c) => { length += c.length; });
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, length * 2, true);

  let offset = 44;
  chunks.forEach((chunk) => {
    for (let i = 0; i < chunk.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, sample * 32767, true);
      offset += 2;
    }
  });
  return new Blob([view], { type: "audio/wav" });
}

/* ---------- voices ---------- */

function fillVoiceSelect(select, voices, selectedId) {
  const groups = new Map();
  voices.forEach((voice) => {
    const label = voice.accent + " " + voice.gender;
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(voice);
  });
  select.innerHTML = "";
  Array.from(groups.keys()).sort().forEach((label) => {
    const group = document.createElement("optgroup");
    group.label = label;
    groups.get(label).forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = voice.name + " (" + voice.grade + ")";
      if (voice.id === selectedId) option.selected = true;
      group.appendChild(option);
    });
    select.appendChild(group);
  });
}

async function loadVoices() {
  const resp = await api("/voices");
  if (!resp.ok) {
    setStatus(await errorMessage(resp), true);
    return;
  }
  const body = await resp.json();
  const saved = localStorage.getItem("kokoro.voice");
  fillVoiceSelect($("#voice"), body.voices, saved || body.default);
  fillVoiceSelect($("#voiceB"), body.voices, "af_sky");
}

async function loadHealth() {
  try {
    const resp = await fetch("/health");
    const body = await resp.json();
    $("#badge").textContent =
      body.status === "ok"
        ? body.device + " · warmup " + Number(body.warmup_seconds || 0).toFixed(1) + "s"
        : "model loading…";
  } catch (err) {
    $("#badge").textContent = "server unreachable";
  }
}

function voiceSpec() {
  const primary = $("#voice").value;
  if (!$("#blendOn").checked) return primary;
  const mix = Number($("#mix").value) / 100;
  return primary + ":" + (1 - mix).toFixed(2) + "," + $("#voiceB").value + ":" + mix.toFixed(2);
}

function requestBody() {
  return {
    text: $("#text").value,
    voice: voiceSpec(),
    lang: $("#lang").value || null,
    speed: Number($("#speed").value),
    format: $("#format").value,
  };
}

/* ---------- highlighting ---------- */

function findWordIndex(words, seconds) {
  let low = 0;
  let high = words.length - 1;
  let found = -1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (words[mid].start <= seconds) {
      found = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  if (found >= 0 && seconds > words[found].end + 0.15) return -1;
  return found;
}

function appendWord(word) {
  const span = document.createElement("span");
  span.textContent = word.word;
  $("#transcript").appendChild(span);
  $("#transcript").appendChild(document.createTextNode(" "));
}

function startHighlighting(clock) {
  cancelAnimationFrame(state.raf);
  const tick = () => {
    const index = findWordIndex(state.words, clock());
    const spans = $("#transcript").querySelectorAll("span");
    for (let i = 0; i < spans.length; i += 1) {
      spans[i].classList.toggle("active", i === index);
    }
    if (state.playing) state.raf = requestAnimationFrame(tick);
  };
  state.raf = requestAnimationFrame(tick);
}

/* ---------- playback ---------- */

function ensureContext() {
  if (!state.ctx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    state.ctx = new Ctor();
  }
  if (state.ctx.state === "suspended") state.ctx.resume();
  return state.ctx;
}

function scheduleChunk(event) {
  const samples = decodePcm(event.audio);
  state.chunks.push(samples);

  const ctx = state.ctx;
  const buffer = ctx.createBuffer(1, samples.length, state.sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);

  if (state.scheduleAt === 0) {
    // Small lead-in so the first chunk is not clipped by scheduling latency.
    state.scheduleAt = ctx.currentTime + 0.12;
    state.playStart = state.scheduleAt;
    state.playing = true;
    startHighlighting(() => state.ctx.currentTime - state.playStart);
  }
  source.start(state.scheduleAt);
  state.scheduleAt += buffer.duration;
  state.sources.push(source);

  (event.words || []).forEach((word) => {
    state.words.push(word);
    appendWord(word);
  });
}

function stop() {
  state.playing = false;
  cancelAnimationFrame(state.raf);
  state.sources.forEach((source) => {
    try { source.stop(); } catch (err) { /* already finished */ }
  });
  state.sources = [];
  state.scheduleAt = 0;
  const player = $("#player");
  player.pause();
  $("#transcript").querySelectorAll("span.active").forEach((span) => {
    span.classList.remove("active");
  });
}

function offerDownload(blob, filename) {
  const link = $("#download");
  if (link.dataset.url) URL.revokeObjectURL(link.dataset.url);
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.dataset.url = url;
  link.download = filename;
  link.hidden = false;
}

/* ---------- synthesis ---------- */

async function speak() {
  const text = $("#text").value.trim();
  if (!text) {
    setStatus("Type something first.", true);
    return;
  }

  stop();
  state.words = [];
  state.chunks = [];
  $("#transcript").innerHTML = "";
  $("#download").hidden = true;
  $("#speak").disabled = true;
  localStorage.setItem("kokoro.voice", $("#voice").value);
  localStorage.setItem("kokoro.key", $("#apiKey").value);

  const started = performance.now();
  let firstAudioAt = null;
  setStatus("Synthesizing…");

  let resp;
  try {
    resp = await api("/tts/stream", {
      method: "POST",
      body: JSON.stringify(requestBody()),
    });
  } catch (err) {
    $("#speak").disabled = false;
    return fallback("stream request failed");
  }

  if (!resp.ok) {
    setStatus(await errorMessage(resp), true);
    $("#speak").disabled = false;
    return;
  }
  if (!resp.body || !(window.AudioContext || window.webkitAudioContext)) {
    $("#speak").disabled = false;
    return fallback("streaming unsupported in this browser");
  }

  ensureContext();
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      const lines = buffered.split("\n");
      buffered = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "meta") {
          state.sampleRate = event.sample_rate;
        } else if (event.type === "chunk") {
          if (firstAudioAt === null) firstAudioAt = performance.now();
          scheduleChunk(event);
        } else if (event.type === "error") {
          setStatus("Synthesis failed: " + event.message, true);
          $("#speak").disabled = false;
          return;
        } else if (event.type === "done") {
          reportStats(started, firstAudioAt, event.duration);
        }
      }
    }
  } catch (err) {
    setStatus("Stream interrupted: " + err, true);
  }

  $("#speak").disabled = false;
  if (state.chunks.length) {
    offerDownload(wavBlob(state.chunks, state.sampleRate), "kokoro.wav");
    const tail = (state.scheduleAt - state.ctx.currentTime + 0.4) * 1000;
    setTimeout(() => { state.playing = false; }, Math.max(0, tail));
  }
}

function reportStats(started, firstAudioAt, audioDuration) {
  const total = (performance.now() - started) / 1000;
  const ttfa = firstAudioAt ? (firstAudioAt - started) / 1000 : total;
  const rtf = audioDuration ? total / audioDuration : 0;
  setStatus(
    "audio " + audioDuration.toFixed(2) + "s · first sound " + ttfa.toFixed(2) +
    "s · total " + total.toFixed(2) + "s · RTF " + rtf.toFixed(2)
  );
}

/** Non-streaming path: one /tts call, plain <audio> playback. */
async function fallback(reason) {
  setStatus("Falling back to non-streaming mode (" + reason + ")…");
  const started = performance.now();
  const body = Object.assign(requestBody(), { include_timestamps: true });

  let resp;
  try {
    resp = await api("/tts", { method: "POST", body: JSON.stringify(body) });
  } catch (err) {
    setStatus("Request failed: " + err, true);
    return;
  }
  if (!resp.ok) {
    setStatus(await errorMessage(resp), true);
    return;
  }

  const payload = await resp.json();
  state.words = payload.words || [];
  $("#transcript").innerHTML = "";
  state.words.forEach(appendWord);

  const player = $("#player");
  const mime = payload.format === "mp3" ? "audio/mpeg" : "audio/wav";
  player.src = "data:" + mime + ";base64," + payload.audio;
  player.hidden = false;
  state.playing = true;
  startHighlighting(() => player.currentTime);
  player.onended = () => { state.playing = false; };
  await player.play();

  const bytes = atob(payload.audio);
  const array = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i += 1) array[i] = bytes.charCodeAt(i);
  offerDownload(new Blob([array], { type: mime }), "kokoro." + payload.format);
  reportStats(started, performance.now(), payload.duration);
}

/* ---------- wiring ---------- */

function init() {
  $("#apiKey").value = localStorage.getItem("kokoro.key") || "";
  $("#text").value = "Kokoro is an open weight text to speech model with eighty two million parameters.";
  $("#count").textContent = $("#text").value.length;

  $("#text").addEventListener("input", () => {
    $("#count").textContent = $("#text").value.length;
  });
  $("#speed").addEventListener("input", () => {
    $("#speedOut").textContent = Number($("#speed").value).toFixed(2);
  });
  $("#mix").addEventListener("input", () => {
    $("#mixOut").textContent = $("#mix").value + "%";
  });
  $("#speak").addEventListener("click", speak);
  $("#stop").addEventListener("click", stop);
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") speak();
  });

  loadVoices();
  loadHealth();
}

document.addEventListener("DOMContentLoaded", init);
