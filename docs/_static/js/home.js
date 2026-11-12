// A leaky reservoir driven by a slow signal, drawn as a raster of its states.
(() => {
  const canvas = document.querySelector(".td-states");
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

  const units = 64;
  const density = 0.1;
  const leak = 0.25;
  // By the circular law, a sparse uniform matrix has a spectral radius of
  // about sqrt(units * density / 3); scale it to 0.95.
  const scale = 0.95 / Math.sqrt((units * density) / 3);
  const recurrent = Array.from({ length: units }, () =>
    Array.from({ length: units }, () => (rand() < density ? uniform() * scale : 0)),
  );
  const inputWeights = Array.from({ length: units }, uniform);
  const frequency = 0.11;
  const input = (t) => Math.sin(t * frequency) * (0.7 + 0.3 * Math.sin(t * 0.017));

  let state = new Float64Array(units);
  let time = 0;
  const step = () => {
    const u = input(time++);
    const next = new Float64Array(units);
    for (let i = 0; i < units; i++) {
      let pre = inputWeights[i] * u;
      for (let j = 0; j < units; j++) pre += recurrent[i][j] * state[j];
      next[i] = (1 - leak) * state[i] + leak * Math.tanh(pre);
    }
    state = next;
    return { u, h: next };
  };

  // Order the rows by each unit's phase against the input, so the response
  // reads as bands rather than noise.
  const sin = new Float64Array(units);
  const cos = new Float64Array(units);
  for (let k = 0; k < 600; k++) {
    const { h } = step();
    for (let i = 0; i < units; i++) {
      sin[i] += h[i] * Math.sin(time * frequency);
      cos[i] += h[i] * Math.cos(time * frequency);
    }
  }
  const order = [...Array(units).keys()].sort(
    (a, b) => Math.atan2(sin[a], cos[a]) - Math.atan2(sin[b], cos[b]),
  );

  const columns = 160;
  const history = Array.from({ length: columns }, step);
  const raster = document.createElement("canvas");
  raster.width = columns;
  raster.height = units;
  const pixels = raster.getContext("2d").createImageData(columns, units);

  let colours;
  const readColours = () => {
    const style = getComputedStyle(document.body);
    const rgb = (name) => {
      const hex = style.getPropertyValue(name).trim();
      return [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16));
    };
    colours = {
      high: rgb("--td-state-high"),
      low: rgb("--td-state-low"),
      signal: style.getPropertyValue("--md-default-fg-color--light"),
    };
  };

  const resize = () => {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  };

  const draw = () => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const band = Math.round(height * 0.18);
    ctx.clearRect(0, 0, width, height);

    // the input, on top
    ctx.beginPath();
    history.forEach(({ u }, k) =>
      ctx.lineTo((k / (columns - 1)) * width, band / 2 - u * (band / 2 - 3)),
    );
    ctx.strokeStyle = colours.signal;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // the states, one row per unit, each scaled to its own range
    const peak = new Float64Array(units).fill(1e-3);
    for (const { h } of history) h.forEach((v, i) => (peak[i] = Math.max(peak[i], Math.abs(v))));
    for (let k = 0; k < columns; k++) {
      const { h } = history[k];
      order.forEach((unit, row) => {
        const v = h[unit] / peak[unit];
        const [r, g, b] = v > 0 ? colours.high : colours.low;
        const at = 4 * (row * columns + k);
        pixels.data[at] = r;
        pixels.data[at + 1] = g;
        pixels.data[at + 2] = b;
        pixels.data[at + 3] = Math.round(230 * Math.min(1, Math.abs(v)) ** 0.8);
      });
    }
    raster.getContext("2d").putImageData(pixels, 0, 0);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(raster, 0, band + 8, width, height - band - 8);
  };

  readColours();
  resize();
  draw();
  new ResizeObserver(() => {
    resize();
    draw();
  }).observe(canvas);
  // the palette toggle switches the body's colour scheme
  new MutationObserver(() => {
    readColours();
    draw();
  }).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  let visible = true;
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
  }).observe(canvas);
  let frame = 0;
  const tick = () => {
    // two frames per step keep the bands drifting slowly
    if (visible && frame++ % 2 === 0) {
      history.push(step());
      history.shift();
      draw();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
})();
