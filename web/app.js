"use strict";

const $ = (selector) => document.querySelector(selector);

const state = {
  words: [],
  raf: 0,
  objectUrl: null,
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

function decodeBase64(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
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

function clearHighlight() {
  $("#transcript").querySelectorAll("span.active").forEach((span) => {
    span.classList.remove("active");
  });
}

function startHighlighting() {
  const player = $("#player");
  cancelAnimationFrame(state.raf);
  const tick = () => {
    const index = findWordIndex(state.words, player.currentTime);
    const spans = $("#transcript").querySelectorAll("span");
    for (let i = 0; i < spans.length; i += 1) {
      spans[i].classList.toggle("active", i === index);
    }
    if (!player.paused && !player.ended) state.raf = requestAnimationFrame(tick);
  };
  state.raf = requestAnimationFrame(tick);
}

function stopHighlighting() {
  cancelAnimationFrame(state.raf);
}

/* ---------- playback ---------- */

function loadAudio(bytes, format) {
  const mime = format === "mp3" ? "audio/mpeg" : "audio/wav";
  const blob = new Blob([bytes], { type: mime });
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  const url = URL.createObjectURL(blob);
  state.objectUrl = url;

  $("#player").src = url;

  const link = $("#download");
  link.href = url;
  link.download = "kokoro." + format;
  link.hidden = false;
}

function stop() {
  const player = $("#player");
  player.pause();
  player.currentTime = 0;
  stopHighlighting();
  clearHighlight();
}

/* ---------- synthesis ---------- */

function reportStats(started, audioDuration) {
  const total = (performance.now() - started) / 1000;
  const rtf = audioDuration ? total / audioDuration : 0;
  return (
    "audio " + audioDuration.toFixed(2) + "s · synth " + total.toFixed(2) +
    "s · RTF " + rtf.toFixed(2)
  );
}

async function speak() {
  const text = $("#text").value.trim();
  if (!text) {
    setStatus("Type something first.", true);
    return;
  }

  stop();
  state.words = [];
  $("#transcript").innerHTML = "";
  $("#download").hidden = true;
  $("#speak").disabled = true;
  localStorage.setItem("kokoro.voice", $("#voice").value);
  localStorage.setItem("kokoro.key", $("#apiKey").value);

  const started = performance.now();
  setStatus("Synthesizing…");

  const body = Object.assign(requestBody(), { include_timestamps: true });
  let resp;
  try {
    resp = await api("/tts", { method: "POST", body: JSON.stringify(body) });
  } catch (err) {
    setStatus("Request failed: " + err, true);
    $("#speak").disabled = false;
    return;
  }

  if (!resp.ok) {
    setStatus(await errorMessage(resp), true);
    $("#speak").disabled = false;
    return;
  }

  const payload = await resp.json();
  $("#speak").disabled = false;

  state.words = payload.words || [];
  $("#transcript").innerHTML = "";
  state.words.forEach(appendWord);

  loadAudio(decodeBase64(payload.audio), payload.format);

  const statsLine = reportStats(started, payload.duration);
  const player = $("#player");
  try {
    await player.play();
    setStatus(statsLine);
  } catch (err) {
    setStatus(statsLine + " — press play to listen (autoplay was blocked).", true);
  }
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

  const player = $("#player");
  player.addEventListener("play", startHighlighting);
  player.addEventListener("pause", stopHighlighting);
  player.addEventListener("ended", () => {
    stopHighlighting();
    clearHighlight();
  });

  loadVoices();
  loadHealth();
}

document.addEventListener("DOMContentLoaded", init);
