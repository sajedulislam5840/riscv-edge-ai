(function () {
  "use strict";
  const T = window.CYCLE_TRACE;
  const root = document.getElementById("trace");
  if (!T || !root) return;

  const css = getComputedStyle(document.documentElement);
  const col = (v) => css.getPropertyValue(v).trim();
  const CLASSES = [
    { k: "L", name: "loading from memory", color: col("--c3") || "#C97C0E" },
    { k: "M", name: "multiplying", color: col("--c1") || "#2F4BD6" },
    { k: "X", name: "simd_mac", color: col("--c5") || "#C7325F" },
    { k: "A", name: "other arithmetic", color: "#8B919C" },
    { k: "B", name: "branches", color: col("--c2") || "#7B3FE4" },
  ];
  const COLOR = Object.fromEntries(CLASSES.map((c) => [c.k, c.color]));
  const fmt = (n) => Math.round(n).toLocaleString("en");

  function prepare(data) {
    const n = data.pc.length;
    const start = new Float64Array(n + 1);
    const cum = {};
    for (const c of CLASSES) cum[c.k] = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
      start[i + 1] = start[i] + data.c[i];
      for (const c of CLASSES) cum[c.k][i + 1] = cum[c.k][i] + (data.cls[i] === c.k ? data.c[i] : 0);
    }
    return { data, n, start, cum, total: start[n] };
  }

  const lanes = [
    { id: "tr-lane-scalar", ...prepare(T.scalar) },
    { id: "tr-lane-mac", ...prepare(T.mac) },
  ];
  const span = Math.max(...lanes.map((l) => l.total));

  for (const lane of lanes) {
    const el = document.getElementById(lane.id);
    lane.canvas = el.querySelector('[data-tr="canvas"]');
    lane.ctx = lane.canvas.getContext("2d");
    lane.strip = el.querySelector('[data-tr="strip"]');
    lane.clock = el.querySelector('[data-tr="clock"]');
    lane.status = el.querySelector('[data-tr="status"]');
    const legend = el.querySelector('[data-tr="legend"]');
    lane.tallyEls = {};
    for (const c of CLASSES) {
      const item = document.createElement("span");
      item.innerHTML = `<i style="background:${c.color}"></i>${c.name} <b>0</b>`;
      legend.appendChild(item);
      lane.tallyEls[c.k] = item.querySelector("b");
    }
    lane.lastIdx = -2;
  }

  const axis = document.getElementById("tr-axis");
  const ticks = [0, 5000, 10000, 15000, span];
  axis.innerHTML = ticks.map((t, i) => `<span style="left:${(100 * t / span).toFixed(2)}%">${fmt(t)}${i === ticks.length - 1 ? " cycles" : ""}</span>`).join("");

  function idxAt(lane, t) {
    if (t >= lane.total) return lane.n;
    let lo = 0, hi = lane.n - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (lane.start[mid] <= t) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  function buildImage(lane) {
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(200, Math.round(lane.canvas.clientWidth * dpr));
    const H = Math.round(lane.canvas.clientHeight * dpr) || 64;
    lane.canvas.width = W; lane.canvas.height = H;
    const colSpan = span / W;
    const cols = {};
    for (const c of CLASSES) cols[c.k] = new Float64Array(W + 1);
    for (let i = 0; i < lane.n; i++) {
      const k = lane.data.cls[i];
      let s = lane.start[i], e = s + lane.data.c[i];
      while (s < e) {
        const x = Math.min(W - 1, Math.floor(s / colSpan));
        const edge = (x + 1) * colSpan;
        const part = Math.min(e, edge) - s;
        cols[k][x] += part;
        s += part;
        if (part <= 0) break;
      }
    }
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const o = off.getContext("2d");
    for (let x = 0; x < W; x++) {
      let y = H;
      for (const c of CLASSES) {
        const h = (cols[c.k][x] / colSpan) * H;
        if (h <= 0) continue;
        o.fillStyle = c.color;
        o.fillRect(x, y - h, 1, h);
        y -= h;
      }
    }
    lane.off = off; lane.W = W; lane.H = H;
  }

  function drawLane(lane, t) {
    const { ctx, W, H } = lane;
    ctx.clearRect(0, 0, W, H);
    ctx.globalAlpha = 0.14;
    ctx.drawImage(lane.off, 0, 0);
    ctx.globalAlpha = 1;
    const xc = Math.min(t, lane.total) / span * W;
    if (xc > 0) ctx.drawImage(lane.off, 0, 0, Math.max(1, xc), H, 0, 0, Math.max(1, xc), H);
    if (t < lane.total) {
      ctx.fillStyle = "#171A21";
      ctx.fillRect(Math.min(W - 2, xc), 0, 2, H);
    }
  }

  function updateText(lane, t) {
    const done = t >= lane.total;
    const shown = Math.min(t, lane.total);
    lane.clock.textContent = fmt(shown);
    lane.status.textContent = t <= 0 ? "waiting" : done ? "finished" : "running";
    const i = idxAt(lane, t);
    for (const c of CLASSES) {
      const v = done ? lane.cum[c.k][lane.n] : lane.cum[c.k][i] + (lane.data.cls[i] === c.k ? t - lane.start[i] : 0);
      lane.tallyEls[c.k].textContent = fmt(v);
    }
    if (i === lane.lastIdx) return;
    lane.lastIdx = i;
    if (done) {
      lane.strip.innerHTML = `<span class="tr-strip-note">Finished after ${fmt(lane.total)} cycles and ${fmt(lane.n)} instructions.</span>`;
      return;
    }
    let html = "";
    const from = Math.max(0, i - 2), to = Math.min(lane.n, i + 12);
    for (let j = from; j < to; j++) {
      const full = T.names[lane.data.op[j]];
      const name = full.startsWith("simd_") ? full.slice(5) : full;
      html += `<span class="tr-chip${j === i ? " is-now" : ""}" style="flex-grow:${lane.data.c[j]};background:${COLOR[lane.data.cls[j]]}" title="${full}, 0x${lane.data.pc[j].toString(16)}, ${lane.data.c[j]} cycles">${name}</span>`;
    }
    lane.strip.innerHTML = html;
  }

  const playBtn = document.getElementById("tr-play");
  const restartBtn = document.getElementById("tr-restart");
  const speedSel = document.getElementById("tr-speed");
  const scrub = document.getElementById("tr-scrub");
  scrub.max = String(span);
  let t = 0, playing = false, last = 0, startedOnce = false;

  function render() {
    for (const lane of lanes) { drawLane(lane, t); updateText(lane, t); }
    scrub.value = String(Math.round(t));
    playBtn.textContent = playing ? "Pause" : t >= span ? "Replay" : "Play";
  }

  function frame(now) {
    if (!playing) return;
    t = Math.min(span, t + ((now - last) / 1000) * Number(speedSel.value));
    last = now;
    if (t >= span) playing = false;
    render();
    if (playing) requestAnimationFrame(frame);
  }

  function play() {
    if (t >= span) t = 0;
    playing = true; startedOnce = true; last = performance.now();
    requestAnimationFrame(frame);
    render();
  }

  playBtn.addEventListener("click", () => { if (playing) { playing = false; render(); } else play(); });
  restartBtn.addEventListener("click", () => { t = 0; for (const l of lanes) l.lastIdx = -2; render(); if (!playing) play(); });
  scrub.addEventListener("input", () => { playing = false; t = Number(scrub.value); for (const l of lanes) l.lastIdx = -2; render(); });

  const [sc, mc] = lanes;
  const loadShare = (mc.cum.L[mc.n] / mc.total) * 100;
  const xShare = (mc.cum.X[mc.n] / mc.total) * 100;
  document.getElementById("tr-summary").textContent =
    `One neuron's dot product, ${T.meta.macs} multiply-accumulates: ${fmt(sc.total)} cycles in the scalar loop, ${fmt(mc.total)} with the custom instruction, ` +
    `${(sc.total / mc.total).toFixed(2)}x fewer. The ${fmt(mc.data.cls.split("X").length - 1)} simd_mac instructions take ${xShare.toFixed(0)}% of the MAC version's cycles; ` +
    `${loadShare.toFixed(0)}% goes on loading their operands from memory. That is where the next gain would come from.`;

  function layout() {
    for (const lane of lanes) buildImage(lane);
    render();
  }
  let resizeTimer = 0;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(layout, 150); });
  layout();

  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) {
    t = span; render();
  } else if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting) && !startedOnce) { play(); io.disconnect(); }
    }, { threshold: 0.35 });
    io.observe(root);
  }
})();
