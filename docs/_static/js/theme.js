// Follow the light or dark palette chosen on the documentation pages.
try {
  const key = new URL(".", location).pathname + ".__palette";
  const scheme = JSON.parse(localStorage.getItem(key))?.color?.scheme;
  if (scheme === "default") document.documentElement.dataset.theme = "light";
  if (scheme === "slate") document.documentElement.dataset.theme = "dark";
} catch (e) {}
