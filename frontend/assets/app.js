/* Shared frontend helpers. Vanilla JS, no build step, no dependencies. */

const API = "";  // same origin

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

function severityOf(priority) {
  if (priority >= 75) return "critical";
  if (priority >= 50) return "high";
  if (priority >= 25) return "medium";
  return "low";
}

function fmtTime(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch (e) { return iso; }
}

function ago(iso) {
  const secs = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (secs < 60) return `${Math.floor(secs)}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

let _toastTimer = null;
function toast(msg) {
  let t = document.querySelector(".toast");
  if (!t) { t = el("div", { class: "toast" }); document.body.append(t); }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}

/* WebSocket with auto-reconnect — a dropped venue connection must self-heal. */
function connectWS(onMessage, onStatus) {
  let ws, retry = 0, closed = false;
  function open() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => { retry = 0; onStatus && onStatus(true); };
    ws.onmessage = (ev) => { try { onMessage(JSON.parse(ev.data)); } catch (e) {} };
    ws.onclose = () => {
      onStatus && onStatus(false);
      if (closed) return;
      retry = Math.min(retry + 1, 6);
      setTimeout(open, retry * 500);
    };
    ws.onerror = () => ws.close();
  }
  open();
  return { close() { closed = true; ws && ws.close(); }, send(m) { ws && ws.readyState === 1 && ws.send(m); } };
}

/* Render the shared campus map into an <svg> from /api/zones.
   Zone polygons are normalised [0,1]; we scale to a 960x540 viewBox so the map
   and the backend's point-in-polygon tests agree by construction. */
const MAP_W = 960, MAP_H = 540;
const SEV_COLOR = { critical: "#ff4d5e", high: "#ff9f40", medium: "#ffd23f", low: "#4dd4ac" };

function renderMap(svg, zones, opts = {}) {
  const NS = "http://www.w3.org/2000/svg";
  svg.setAttribute("viewBox", `0 0 ${MAP_W} ${MAP_H}`);
  svg.setAttribute("class", "map");
  svg.innerHTML = "";

  // Backdrop: paths/lawns so the map reads as a campus, not a grid.
  const bg = document.createElementNS(NS, "rect");
  bg.setAttribute("x", 0); bg.setAttribute("y", 0);
  bg.setAttribute("width", MAP_W); bg.setAttribute("height", MAP_H);
  bg.setAttribute("fill", "#0c1116");
  svg.append(bg);

  // Simple walkway cross to suggest campus roads.
  for (const [x1, y1, x2, y2] of [[0, 300, 960, 300], [480, 0, 480, 540]]) {
    const road = document.createElementNS(NS, "line");
    road.setAttribute("x1", x1); road.setAttribute("y1", y1);
    road.setAttribute("x2", x2); road.setAttribute("y2", y2);
    road.setAttribute("stroke", "#1a222c"); road.setAttribute("stroke-width", 16);
    svg.append(road);
  }

  for (const z of zones) {
    const pts = z.polygon.map(([x, y]) => `${(x * MAP_W).toFixed(1)},${(y * MAP_H).toFixed(1)}`).join(" ");
    const risk = z.zone_risk ?? z.risk_prior ?? 0.15;
    const poly = document.createElementNS(NS, "polygon");
    poly.setAttribute("points", pts);
    poly.setAttribute("class", "zone");
    // Fill intensity tracks zone risk; stroke marks open incidents.
    const open = z.open_incidents || 0;
    poly.setAttribute("fill", open > 0 ? "rgba(255,77,94,0.18)" : `rgba(77,163,255,${0.06 + risk * 0.22})`);
    poly.setAttribute("stroke", open > 0 ? SEV_COLOR.critical : "#2d3a4a");
    poly.setAttribute("stroke-width", open > 0 ? 2.5 : 1.2);
    if (opts.onZoneClick) poly.addEventListener("click", () => opts.onZoneClick(z));
    svg.append(poly);

    const [cx, cy] = z.centroid || [
      z.polygon.reduce((a, p) => a + p[0], 0) / z.polygon.length,
      z.polygon.reduce((a, p) => a + p[1], 0) / z.polygon.length,
    ];
    const label = document.createElementNS(NS, "text");
    label.setAttribute("x", cx * MAP_W); label.setAttribute("y", cy * MAP_H);
    label.setAttribute("class", "zone-label");
    label.textContent = z.name;
    svg.append(label);

    if (z.incidents_30d != null) {
      const sub = document.createElementNS(NS, "text");
      sub.setAttribute("x", cx * MAP_W); sub.setAttribute("y", cy * MAP_H + 15);
      sub.setAttribute("class", "zone-count");
      sub.textContent = open > 0 ? `${open} open · ${z.incidents_30d}/30d` : `${z.incidents_30d} in 30d`;
      svg.append(sub);
    }
  }
  return svg;
}

function pinOnMap(svg, zone, severity) {
  const NS = "http://www.w3.org/2000/svg";
  const [cx, cy] = zone.centroid || [0.5, 0.5];
  const g = document.createElementNS(NS, "g");
  g.setAttribute("class", "incident-pin");
  const c = document.createElementNS(NS, "circle");
  c.setAttribute("cx", cx * MAP_W); c.setAttribute("cy", cy * MAP_H);
  c.setAttribute("r", 10); c.setAttribute("fill", SEV_COLOR[severity] || "#fff");
  c.setAttribute("opacity", 0.95);
  const ring = document.createElementNS(NS, "circle");
  ring.setAttribute("cx", cx * MAP_W); ring.setAttribute("cy", cy * MAP_H);
  ring.setAttribute("r", 10); ring.setAttribute("fill", "none");
  ring.setAttribute("stroke", SEV_COLOR[severity] || "#fff"); ring.setAttribute("stroke-width", 2);
  const anim = document.createElementNS(NS, "animate");
  anim.setAttribute("attributeName", "r"); anim.setAttribute("from", 10); anim.setAttribute("to", 34);
  anim.setAttribute("dur", "1.4s"); anim.setAttribute("repeatCount", "3");
  const fade = document.createElementNS(NS, "animate");
  fade.setAttribute("attributeName", "opacity"); fade.setAttribute("from", 0.9); fade.setAttribute("to", 0);
  fade.setAttribute("dur", "1.4s"); fade.setAttribute("repeatCount", "3");
  ring.append(anim); ring.append(fade);
  g.append(ring); g.append(c);
  svg.append(g);
  setTimeout(() => g.remove(), 4400);
}

/* Contribution-bar block for an incident's signals (the "why"). */
function renderBars(signals, maxWeightSum) {
  const wrap = el("div", { class: "bars" });
  const active = signals.filter((s) => s.contribution > 0).sort((a, b) => b.contribution - a.contribution);
  if (!active.length) return el("div", { class: "small muted" }, "No contributing signals.");
  const max = Math.max(...active.map((s) => s.contribution), 1);
  for (const s of active) {
    wrap.append(
      el("div", { class: "bar-row" },
        el("span", {}, s.label),
        el("div", { class: "bar-track" }, el("div", { class: "bar-fill", style: `width:${(s.contribution / max) * 100}%` })),
        el("span", { class: "bar-val" }, `+${Math.round(s.contribution)}`)
      )
    );
  }
  return wrap;
}
