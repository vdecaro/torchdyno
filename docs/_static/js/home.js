// A leaky reservoir driven by a slow signal, drawn as one crisp row per unit.
(() => {
  const canvas = document.querySelector(".td-states");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let seed = 7;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const uniform = () => 2 * rand() - 1;

  const units = 32;
  const density = 0.15;
  const leak = 0.3;
  const scale = 0.95 / Math.sqrt((units * density) / 3);
  const recurrent = Array.from({ length: units }, () =>
    Array.from({ length: units }, () => (rand() < density ? uniform() * scale : 0)),
  );
  const inputWeights = Array.from({ length: units }, () => uniform() * 1.2);
  const frequency = 0.1;
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

  // order the rows by each unit's phase against the input
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

  const columns = 120;
  const history = Array.from({ length: columns }, step);

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
      label: style.getPropertyValue("--md-default-fg-color--light"),
      font: getComputedStyle(document.body).fontFamily,
    };
  };

  const resize = () => {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  };

  // one shared scale: strong units saturate, quiet ones fade
  const cell = (v) => {
    const [r, g, b] = v > 0 ? colours.high : colours.low;
    return `rgba(${r},${g},${b},${Math.min(1, Math.abs(v) / 0.6) ** 0.75})`;
  };

  const draw = () => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    const label = 16;
    const gap = 1;
    const rowHeight = (height - 2 * label - 10) / (units + 1);
    const colWidth = width / columns;

    ctx.font = `0.62rem ${colours.font}`;
    ctx.fillStyle = colours.label;
    ctx.textBaseline = "top";
    ctx.fillText("input", 0, 0);
    ctx.fillText(`${units} units, ordered by phase`, 0, label + rowHeight + 8);
    ctx.textAlign = "right";
    ctx.fillText("time →", width, height - label + 4);
    ctx.textAlign = "left";

    // whole-pixel cells, so translucent neighbours never overlap
    history.forEach(({ u, h }, k) => {
      const x = Math.round(k * colWidth);
      const w = Math.round((k + 1) * colWidth) - x;
      ctx.fillStyle = cell(u);
      ctx.fillRect(x, label, w, Math.round(rowHeight) - gap);
      order.forEach((unit, row) => {
        ctx.fillStyle = cell(h[unit]);
        ctx.fillRect(x, Math.round(2 * label + 10 + row * rowHeight), w, Math.round(rowHeight) - gap);
      });
    });
  };

  readColours();
  resize();
  draw();
  new ResizeObserver(() => {
    resize();
    draw();
  }).observe(canvas);
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
    if (visible && frame++ % 3 === 0) {
      history.push(step());
      history.shift();
      draw();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
})();
