// A small leaky reservoir driven by a modulated sine, drawn as it runs.
(() => {
  const canvas = document.querySelector(".reservoir");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // A seeded generator, so every visit shows the same reservoir.
  let seed = 7;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const uniform = () => 2 * rand() - 1;

  const units = 60;
  const density = 0.1;
  const leak = 0.3;
  // By the circular law, a sparse uniform matrix has a spectral radius of
  // about sqrt(units * density / 3); scale it to 0.95.
  const scale = 0.95 / Math.sqrt((units * density) / 3);
  const recurrent = Array.from({ length: units }, () =>
    Array.from({ length: units }, () => (rand() < density ? uniform() * scale : 0)),
  );
  const inputWeights = Array.from({ length: units }, uniform);
  const shown = [0, 1, 2, 3, 4, 5];

  let state = new Float64Array(units);
  let time = 0;
  const input = (t) => 0.9 * Math.sin(t * 0.07) * Math.sin(t * 0.013 + 1);

  const step = () => {
    const u = input(time++);
    const next = new Float64Array(units);
    for (let i = 0; i < units; i++) {
      let pre = inputWeights[i] * u;
      for (let j = 0; j < units; j++) pre += recurrent[i][j] * state[j];
      next[i] = (1 - leak) * state[i] + leak * Math.tanh(pre);
    }
    state = next;
    return [u, ...shown.map((i) => state[i])];
  };

  const spacing = 2;
  let history = [];
  let length = 0;

  const resize = () => {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    length = Math.ceil(canvas.clientWidth / spacing) + 1;
    while (history.length < length) history.push(step());
    history = history.slice(-length);
  };

  const hex = (name) => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return [1, 3, 5].map((k) => parseInt(value.slice(k, k + 2), 16));
  };
  const mix = (a, b, w) => `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * w)).join(",")})`;

  const trace = (series, centre, amplitude, colour, width) => {
    ctx.beginPath();
    series.forEach((v, k) => ctx.lineTo(k * spacing, centre - v * amplitude));
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.stroke();
  };

  const draw = () => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    ctx.lineJoin = "round";
    const band = height * 0.22;
    trace(
      history.map((row) => row[0]),
      band / 2 + 4,
      band / 2,
      getComputedStyle(document.documentElement).getPropertyValue("--input"),
      1.5,
    );
    let peak = 1e-3;
    for (const row of history) for (const v of row.slice(1)) peak = Math.max(peak, Math.abs(v));
    const flame = hex("--flame");
    const magenta = hex("--magenta");
    const centre = band + (height - band) / 2;
    const amplitude = ((height - band) / 2 - 6) / peak;
    shown.forEach((_, k) =>
      trace(
        history.map((row) => row[k + 1]),
        centre,
        amplitude,
        mix(flame, magenta, k / (shown.length - 1)),
        2,
      ),
    );
  };

  for (let k = 0; k < 300; k++) step();
  resize();
  draw();
  new ResizeObserver(() => {
    resize();
    draw();
  }).observe(canvas);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  let visible = true;
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
  }).observe(canvas);
  const frame = () => {
    if (visible) {
      history.push(step());
      history.shift();
      draw();
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
})();
