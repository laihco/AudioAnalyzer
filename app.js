import { createFFmpeg, fetchFile } from "https://unpkg.com/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js";

const PALETTES = {
  // matches your Python presets (top -> bottom) :contentReference[oaicite:3]{index=3}
  sunset:   ["#ffcdb2", "#ffb4a2", "#e5989b", "#b5838d", "#6d6875"],
  daytime:  ["#ccd5ae", "#e9edc9", "#fefae0", "#faedcd", "#d4a373"],
  nighttime:["#3a015c", "#4f0147", "#35012c", "#290025", "#11001c"],
  sunrise:  ["#cdb4db", "#ffc8dd", "#ffafcc", "#bde0fe", "#a2d2ff"],
};

const $ = (id) => document.getElementById(id);
const fileEl = $("file");
const sampleEl = $("sample");
const btn = $("btn");
const statusEl = $("status");
const progEl = $("prog");
const cv = $("cv");
const ctx = cv.getContext("2d");
const outVid = $("outVid");
const downloadDiv = $("download");

const gainEl = $("gain");
const smoothEl = $("smoothK");
const shiftEl = $("shiftRange");
const hopEl = $("hop");
const wEl = $("w");
const hEl = $("h");
const palEl = $("palette");
const useCloudEl = $("useCloud");

const gainVal = $("gainVal");
const smoothVal = $("smoothVal");
const shiftVal = $("shiftVal");
const hopVal = $("hopVal");
const fpsVal = $("fpsVal");

let audioBlob = null;
let audioName = "input.mp3";

function setStatus(msg) { statusEl.textContent = msg; }
function setProg(v) { progEl.value = Math.max(0, Math.min(1, v)); }

function hexToRgb(hex) {
  const s = hex.replace("#", "").trim();
  const r = parseInt(s.slice(0,2), 16);
  const g = parseInt(s.slice(2,4), 16);
  const b = parseInt(s.slice(4,6), 16);
  return [r, g, b];
}

function smoothMovingAvg(x, k) {
  k = Math.max(1, Math.floor(k));
  if (k % 2 === 0) k += 1;
  if (k <= 1) return x.slice();
  const out = new Float32Array(x.length);
  const half = Math.floor(k/2);
  for (let i=0; i<x.length; i++) {
    let acc = 0, cnt = 0;
    for (let j=i-half; j<=i+half; j++) {
      const jj = Math.max(0, Math.min(x.length-1, j));
      acc += x[jj]; cnt++;
    }
    out[i] = acc / cnt;
  }
  return out;
}

// Same idea as compute_palette_stop_positions :contentReference[oaicite:4]{index=4}
function computeStops(energy, shiftRange=0.18, minGap=0.04) {
  const e = Math.max(0, Math.min(1, energy));
  const base = [0.0, 0.25, 0.50, 0.75, 1.0];
  const shift = (e - 0.5) * 2.0 * shiftRange;

  const pos = base.slice();
  for (let i=1; i<=3; i++) pos[i] = Math.max(0, Math.min(1, base[i] + shift));

  pos[0] = 0.0;
  for (let i=1; i<5; i++) pos[i] = Math.max(pos[i], pos[i-1] + minGap);
  pos[4] = 1.0;

  for (let i=3; i>=0; i--) pos[i] = Math.min(pos[i], pos[i+1] - minGap);
  pos[0] = 0.0; pos[4] = 1.0;

  for (let i=0; i<5; i++) pos[i] = Math.max(0, Math.min(1, pos[i]));
  return pos;
}

function renderGradientFrame(width, height, paletteHex, stops, cloudImg=null) {
  const colors = paletteHex.map(hexToRgb);
  const img = ctx.createImageData(width, height);
  const data = img.data;

  // For each y, find segment and lerp
  for (let y=0; y<height; y++) {
    const t = y / (height - 1);
    let seg = 0;
    while (seg < 4 && t > stops[seg+1]) seg++;

    const t0 = stops[seg], t1 = stops[seg+1];
    const a = (t - t0) / Math.max(1e-6, (t1 - t0));
    const c0 = colors[seg], c1 = colors[seg+1];

    const r = Math.round(c0[0]*(1-a) + c1[0]*a);
    const g = Math.round(c0[1]*(1-a) + c1[1]*a);
    const b = Math.round(c0[2]*(1-a) + c1[2]*a);

    for (let x=0; x<width; x++) {
      const idx = (y*width + x) * 4;
      data[idx+0] = r;
      data[idx+1] = g;
      data[idx+2] = b;
      data[idx+3] = 255;
    }
  }

  // draw to canvas
  ctx.putImageData(img, 0, 0);

  // optional cloud overlay
  if (cloudImg) {
    ctx.globalAlpha = 0.35;
    ctx.drawImage(cloudImg, 0, -10, width, cloudImg.height * (width / cloudImg.width));
    ctx.globalAlpha = 1.0;
  }
}

async function decodeMp3ToAudioBuffer(blob) {
  const arrayBuf = await blob.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return await audioCtx.decodeAudioData(arrayBuf);
}

function computeRmsEnvelope(audioBuffer, hopSec) {
  const sr = audioBuffer.sampleRate;
  const y = audioBuffer.getChannelData(0); // mono-ish
  const hop = Math.max(1, Math.round(hopSec * sr));
  const win = hop * 2; // simple choice
  const frames = Math.floor((y.length - win) / hop);
  const rms = new Float32Array(Math.max(1, frames));
  for (let i=0; i<rms.length; i++) {
    const start = i * hop;
    let acc = 0;
    for (let j=0; j<win; j++) {
      const s = y[start + j] || 0;
      acc += s*s;
    }
    rms[i] = Math.sqrt(acc / win);
  }

  // normalize 0..1 with percentiles-ish (simple robust)
  const sorted = Array.from(rms).sort((a,b)=>a-b);
  const lo = sorted[Math.floor(sorted.length * 0.10)] ?? 0;
  const hi = sorted[Math.floor(sorted.length * 0.95)] ?? 1;
  const out = new Float32Array(rms.length);
  for (let i=0; i<rms.length; i++) {
    out[i] = Math.max(0, Math.min(1, (rms[i] - lo) / Math.max(1e-6, (hi - lo))));
  }
  return out;
}

async function loadCloudIfNeeded(width) {
  if (!useCloudEl.checked) return null;
  try {
    const img = new Image();
    img.src = "assets/cloud.png";
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
    // scale handled in drawImage
    return img;
  } catch {
    return null;
  }
}

function updateUiReadouts() {
  gainVal.textContent = gainEl.value;
  smoothVal.textContent = smoothEl.value;
  shiftVal.textContent = shiftEl.value;
  hopVal.textContent = hopEl.value;
  fpsVal.textContent = (1 / Math.max(1e-6, parseFloat(hopEl.value))).toFixed(2);
}
[gainEl, smoothEl, shiftEl, hopEl].forEach(el => el.addEventListener("input", updateUiReadouts));
updateUiReadouts();

function enableGenerateIfReady() {
  btn.disabled = !audioBlob;
  if (audioBlob) setStatus("Ready. Click Generate MP4.");
}
fileEl.addEventListener("change", () => {
  const f = fileEl.files?.[0];
  if (!f) return;
  audioBlob = f;
  audioName = f.name || "input.mp3";
  sampleEl.value = "";
  enableGenerateIfReady();
});

sampleEl.addEventListener("change", async () => {
  const url = sampleEl.value;
  if (!url) return;
  setStatus("Downloading sample...");
  const r = await fetch(url);
  audioBlob = await r.blob();
  audioName = url.split("/").pop() || "sample.mp3";
  fileEl.value = "";
  enableGenerateIfReady();
});

btn.addEventListener("click", async () => {
  try {
    btn.disabled = true;
    downloadDiv.innerHTML = "";
    outVid.removeAttribute("src");
    outVid.load();

    const width = parseInt(wEl.value, 10);
    const height = parseInt(hEl.value, 10);
    cv.width = width; cv.height = height;

    const hopSec = parseFloat(hopEl.value);
    const fps = 1 / Math.max(1e-6, hopSec);
    const gain = parseFloat(gainEl.value);
    const smoothK = parseInt(smoothEl.value, 10);
    const shiftRange = parseFloat(shiftEl.value);
    const paletteHex = PALETTES[palEl.value] ?? PALETTES.sunset;

    setStatus("Decoding audio...");
    setProg(0.02);
    const audioBuf = await decodeMp3ToAudioBuffer(audioBlob);

    setStatus("Analyzing energy envelope...");
    setProg(0.08);
    let energy = computeRmsEnvelope(audioBuf, hopSec);
    energy = smoothMovingAvg(energy, smoothK);

    // mimic ENERGY_GAIN behavior (centered at 0.5) :contentReference[oaicite:5]{index=5}
    for (let i=0; i<energy.length; i++) {
      const v = (energy[i] - 0.5) * gain + 0.5;
      energy[i] = Math.max(0, Math.min(1, v));
    }

    const cloudImg = await loadCloudIfNeeded(width);

    setStatus("Loading ffmpeg.wasm...");
    setProg(0.12);
    const ffmpeg = createFFmpeg({ log: false });
    await ffmpeg.load();

    // write audio
    setStatus("Preparing files...");
    setProg(0.16);
    ffmpeg.FS("writeFile", "audio.mp3", await fetchFile(audioBlob));

    // render frames
    setStatus("Rendering frames...");
    const total = energy.length;
    const every = Math.max(1, Math.floor(total / 200)); // throttle progress updates
    for (let i=0; i<total; i++) {
      const stops = computeStops(energy[i], shiftRange, 0.04);
      renderGradientFrame(width, height, paletteHex, stops, cloudImg);

      const pngBlob = await new Promise(res => cv.toBlob(res, "image/png"));
      const pngU8 = new Uint8Array(await pngBlob.arrayBuffer());
      const name = `frame_${String(i).padStart(6, "0")}.png`;
      ffmpeg.FS("writeFile", name, pngU8);

      if (i % every === 0) setProg(0.16 + 0.64 * (i / total));
    }

    // encode mp4 (similar to your Python mux step) :contentReference[oaicite:6]{index=6}
    setStatus("Encoding MP4...");
    setProg(0.82);

    const outName = "out.mp4";

    // Try H.264 first, fallback if build lacks libx264
    try {
      await ffmpeg.run(
        "-framerate", String(fps),
        "-i", "frame_%06d.png",
        "-i", "audio.mp3",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        outName
      );
    } catch {
      await ffmpeg.run(
        "-framerate", String(fps),
        "-i", "frame_%06d.png",
        "-i", "audio.mp3",
        "-shortest",
        "-c:v", "mpeg4",
        "-q:v", "4",
        "-c:a", "aac",
        "-b:a", "192k",
        outName
      );
    }

    setStatus("Finalizing...");
    setProg(0.95);

    const outData = ffmpeg.FS("readFile", outName);
    const outBlob = new Blob([outData.buffer], { type: "video/mp4" });
    const outUrl = URL.createObjectURL(outBlob);

    outVid.src = outUrl;

    const a = document.createElement("a");
    a.href = outUrl;
    a.download = "final_gradient.mp4";
    a.textContent = "Download MP4";
    a.style.display = "inline-block";
    a.style.marginTop = "10px";
    downloadDiv.appendChild(a);

    setProg(1.0);
    setStatus("Done.");
  } catch (e) {
    console.error(e);
    setStatus("Error: " + (e?.message || String(e)));
  } finally {
    btn.disabled = false;
  }
});
