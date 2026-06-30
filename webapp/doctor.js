/* Deck doctor — in-browser port of eval/castability.py + eval/deck_doctor.py.
 * Rates a drafted pool's recommended deck on three validated lenses: POWER (deck_value),
 * BUILDABILITY (P(played|pool) — reuses app.js buildSpellsPlayprob + ppMargin), and FUNCTION
 * (the hypergeometric castability model, Karsten-±2 and outcome-validated). Reuses app.js globals
 * (S, buildSpellsPlayprob, ppCardFeat, ppPoolFeats, ppMargin, recommendLands). */

const CO = ["W", "U", "B", "R", "G"];
const MULL = 2;                                  // matches castability.MULL_DISCOUNT

function comb(n, k) {
  if (k < 0 || k > n || n < 0) return 0;
  k = Math.min(k, n - k); let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
}
function hyperAtLeast(k, N, K, n) {              // P(X >= k), hypergeometric upper tail
  if (k <= 0) return 1; if (K < k || n < k) return 0;
  const denom = comb(N, n); if (!denom) return 0;
  let tot = 0; for (let x = k; x <= Math.min(K, n); x++) tot += comb(K, x) * comb(N - K, n - x);
  return tot / denom;
}
function parsePips(mc) {                          // "{2}{U}{U}" -> {pips:{U:2,...}, hybrid:[["W","U"]]}
  const pips = { W: 0, U: 0, B: 0, R: 0, G: 0 }, hybrid = [];
  for (const m of (mc || "").matchAll(/\{([^}]+)\}/g)) {
    const s = m[1].toUpperCase();
    if (s.includes("/")) { const p = s.split("/").filter((c) => CO.includes(c)); if (p.length) hybrid.push(p); }
    else if (CO.includes(s)) pips[s] += 1;
  }
  return { pips, hybrid };
}
function seenCards(turn, deckSize = 40, onPlay = true) {
  return Math.max(1, Math.min(deckSize, 7 + (onPlay ? turn - 1 : turn) - MULL));
}
function pColorOk(deckSize, sources, needed, turn) {
  return hyperAtLeast(needed, deckSize, sources, seenCards(turn, deckSize));
}
function jointCastable(deckSize, seen, needLands, needed, otherLands, nonlands) {
  const denom = comb(deckSize, seen); if (!denom) return 0;
  function rec(idx, drawn, ways) {
    if (idx === needed.length) {
      let tot = 0;
      for (let y = 0; y <= Math.min(otherLands, seen - drawn); y++) {
        const rest = seen - drawn - y;
        if (rest >= 0 && rest <= nonlands && drawn + y >= needLands) tot += ways * comb(otherLands, y) * comb(nonlands, rest);
      }
      return tot;
    }
    const [size, need] = needed[idx]; let s = 0;
    for (let x = need; x <= Math.min(size, seen - drawn); x++) s += rec(idx + 1, drawn + x, ways * comb(size, x));
    return s;
  }
  return rec(0, 0, 1) / denom;
}
function spellCastability(cmc, parsed, deckSize, lands, sources, maxTurn = 7) {
  const turn = Math.min(Math.round(cmc || 0) || 1, maxTurn), seen = seenCards(turn, deckSize);
  const needed = CO.filter((c) => parsed.pips[c] > 0).map((c) => [sources[c] || 0, parsed.pips[c]]).sort((a, b) => a[0] - b[0]);
  const usedSrc = needed.reduce((s, [sc]) => s + sc, 0);
  let p = jointCastable(deckSize, seen, turn, needed, Math.max(0, lands - usedSrc), Math.max(0, deckSize - lands));
  for (const hy of parsed.hybrid) p *= Math.max(...hy.map((c) => pColorOk(deckSize, sources[c] || 0, 1, turn)));
  return p;
}
function inferManabase(deckIdx, nLands) {         // largest-remainder land split by pip demand
  const demand = { W: 0, U: 0, B: 0, R: 0, G: 0 };
  for (const i of deckIdx) { const c = S.cards[i]; if (c.t === "land") continue; const pp = parsePips(c.mc).pips; for (const k of CO) demand[k] += pp[k]; }
  const tot = CO.reduce((s, k) => s + demand[k], 0);
  if (!tot) return { W: 0, U: 0, B: 0, R: 0, G: 0 };
  const raw = {}, alloc = {}; CO.forEach((k) => { raw[k] = (nLands * demand[k]) / tot; alloc[k] = Math.floor(raw[k]); });
  let rem = nLands - CO.reduce((s, k) => s + alloc[k], 0);
  for (const k of CO.slice().sort((a, b) => (raw[b] - alloc[b]) - (raw[a] - alloc[a]))) { if (rem-- <= 0) break; alloc[k] += 1; }
  return alloc;
}
function castabilityScore(deckIdx, nLands, sources) {
  const spells = deckIdx.filter((i) => S.cards[i].t !== "land");
  if (!spells.length) return 0;
  sources = sources || inferManabase(deckIdx, nLands);
  return spells.reduce((s, i) => s + spellCastability(S.cards[i].cmc || 0, parsePips(S.cards[i].mc), 40, nLands, sources), 0) / spells.length;
}
function perTurnCastability(deckIdx, nLands) {
  const sources = inferManabase(deckIdx, nLands), out = {};
  for (let t = 1; t <= 7; t++) {
    const b = deckIdx.filter((i) => S.cards[i].t !== "land" && Math.min(Math.round(S.cards[i].cmc || 0) || 1, 7) === t);
    if (b.length) out[t] = b.reduce((s, i) => s + spellCastability(S.cards[i].cmc || 0, parsePips(S.cards[i].mc), 40, nLands, sources), 0) / b.length;
  }
  return out;
}

// P(played|pool) per candidate (reuses app.js ppCardFeat/ppPoolFeats/ppMargin)
function playProbs(pool, cand) {
  const pf = ppPoolFeats(pool);
  const out = {};
  cand.forEach((i) => { const m = ppMargin([...ppCardFeat(i), ...pf], S.playprob); out[i] = 1 / (1 + Math.exp(-m)); });
  return out;
}
const gradeOf = (x) => (x >= 0.85 ? "A" : x >= 0.72 ? "B" : x >= 0.58 ? "C" : x >= 0.45 ? "D" : "F");

// `fixedDeck` (optional): nonland spell indices WITH multiples to rate AS-IS — the deck the player
// ACTUALLY ran (from the baked deck_ list). When omitted, build the deck from the pool via P(played|pool).
function rateDeck(pool, fixedDeck) {
  const actual = fixedDeck && fixedDeck.length ? fixedDeck.filter((i) => S.cards[i].t !== "land") : null;
  const ranked = actual || buildSpellsPlayprob(pool);             // P(played)-ranked nonland (app.js), or as-run
  const { lands, avgCmc } = recommendLands(ranked.slice(0, 23).map((i) => S.cards[i].cmc || 0));
  const deck = actual ? actual : ranked.slice(0, 40 - lands), deckSet = new Set(deck);
  const dv = deck.map((i) => S.cards[i].deck_value).filter((v) => v != null);
  const powerMean = dv.length ? dv.reduce((a, b) => a + b, 0) / dv.length : 0;
  const setDv = S.cards.filter((c) => c.t !== "land" && c.deck_value != null).map((c) => c.deck_value);
  const powerPct = setDv.length ? setDv.filter((v) => v < powerMean).length / setDv.length : 0.5;
  const cast = castabilityScore(deck, lands), perTurn = perTurnCastability(deck, lands);
  const cc = {}; deck.forEach((i) => (S.cards[i].ci || "").split("").forEach((c) => c !== "C" && (cc[c] = (cc[c] || 0) + 1)));
  const counts = Object.values(cc).sort((a, b) => b - a), tot = counts.reduce((a, b) => a + b, 0);
  const coherence = tot ? counts.slice(0, 2).reduce((a, b) => a + b, 0) / tot : 1;
  const real = Object.entries(cc).filter(([, n]) => n >= 3).sort((a, b) => b[1] - a[1]).map(([c]) => c);
  const overall = 0.4 * powerPct + 0.35 * cast + 0.25 * coherence;

  // advice
  const nonland = [...new Set(pool)].filter((i) => S.cards[i].t !== "land");
  const probs = playProbs(pool, nonland);
  const onColor = (i) => (S.cards[i].ci || "").split("").every((c) => c === "C" || real.includes(c));
  const weakIn = [...new Set(deck)].filter((i) => (probs[i] ?? 1) < 0.5).sort((a, b) => probs[a] - probs[b]).slice(0, 3);
  const strongLeft = nonland.filter((i) => !deckSet.has(i) && onColor(i) && (probs[i] ?? 0) > 0.6).sort((a, b) => probs[b] - probs[a]).slice(0, 3);
  const advice = [];
  if (real.length > 2) advice.push(`${real.length} colors (${real.join("/")}) — consider cutting to the best 2 for consistency.`);
  const weakTurns = Object.entries(perTurn).filter(([, p]) => p < 0.7).sort((a, b) => a[1] - b[1]).slice(0, 3).map(([t]) => t);
  if (weakTurns.length) advice.push(`shaky on-curve castability at turn ${weakTurns.join(", ")} — add colored sources or lower the curve.`);
  weakIn.forEach((i) => advice.push(`borderline include — ${S.cards[i].name} (P=${probs[i].toFixed(2)}): the build model would often cut it.`));
  strongLeft.forEach((i) => advice.push(`consider running — ${S.cards[i].name} (P=${probs[i].toFixed(2)}): left in your sideboard.`));
  if (!advice.length) advice.push("clean build — coherent colors, on-curve, no obvious cuts/adds.");

  return { deck, lands, avgCmc, colors: real, powerMean, powerPct, cast, coherence, perTurn, overall, grade: gradeOf(overall), advice };
}

// ---- UI -----------------------------------------------------------------------------------------
// the built deck, grouped by mana value (a curve view), with each card's color + deck_value
const BASIC_LAND = { W: "Plains", U: "Island", B: "Swamp", R: "Mountain", G: "Forest" };

function deckListHTML(deckIdx, lands) {
  const byCmc = {};                                          // curve view: a row per mana value, card art in it
  deckIdx.forEach((i) => { const m = Math.min(7, Math.round(S.cards[i].cmc || 0)); (byCmc[m] = byCmc[m] || []).push(i); });
  const rows = Object.keys(byCmc).sort((a, b) => a - b).map((m) => {
    const cnt = {}; byCmc[m].forEach((i) => (cnt[i] = (cnt[i] || 0) + 1));   // collapse repeats -> ×N badge
    const uniq = [...new Set(byCmc[m])].sort((a, b) => (S.cards[b].deck_value || 0) - (S.cards[a].deck_value || 0));
    return `<div class="dcrow"><span class="cmc">${m}</span><span class="dcards">` +
    uniq.map((i) => {
      const c = S.cards[i], n = cnt[i];
      const pic = c.img ? `<img class="dcart" loading="lazy" src="${c.img}" alt="${c.name}">` : "";
      return `<span class="dcell">${pic}<span class="dcname">${n > 1 ? `×${n} ` : ""}${c.name}</span></span>`;   // art + exact name
    }).join("") + `</span></div>`;
  }).join("");
  // basic-land split from the deck's colored-pip demand (eval/castability.infer_manabase, in-browser)
  const mb = inferManabase(deckIdx, lands || 17);
  const lh = CO.filter((c) => mb[c] > 0)
    .map((c) => `<span class="lrow"><span class="pip c${c}"></span>${mb[c]} ${BASIC_LAND[c]}</span>`).join("")
    || `${lands || 0} colorless`;
  return `<div class="deckart">${rows}</div><div class="decklands"><b>Lands (${lands || 0}):</b> ${lh}</div>`;
}

// full-screen deck view: a mana-curve of columns (one per CMC), real card art + exact names in each.
// `deckIdx` may carry repeats (a 2-of appears twice) — collapse to one cell per card with a ×N badge;
// the column header count still counts copies, so the curve reflects the real card totals.
function curveColumnsHTML(deckIdx) {
  const byCmc = {};
  deckIdx.forEach((i) => { const m = Math.min(7, Math.round(S.cards[i].cmc || 0)); (byCmc[m] = byCmc[m] || []).push(i); });
  const ms = Object.keys(byCmc).map(Number).sort((a, b) => a - b);
  return `<div class="mvcurve">` + ms.map((m) => {
    const col = byCmc[m], cnt = {};                                  // copies of each card in this column
    col.forEach((i) => (cnt[i] = (cnt[i] || 0) + 1));
    const uniq = [...new Set(col)].sort((a, b) => (S.cards[b].deck_value || 0) - (S.cards[a].deck_value || 0));
    return `<div class="mvcol"><div class="mvhead">${m === 7 ? "7+" : m} <small>(${col.length})</small></div>` +
      uniq.map((i) => { const c = S.cards[i], n = cnt[i];
        return `<div class="mvcard">${c.img ? `<img loading="lazy" src="${c.img}" alt="${c.name}">` : ""}` +
          `<span>${n > 1 ? `<b class="mult">×${n}</b> ` : ""}${colorPips((c.ci || "C").split("").filter((x) => x))}${c.name}</span></div>`;
      }).join("") + `</div>`;
  }).join("") + `</div>`;
}

function deckLandsHTML(deckIdx, lands) {            // exact basic-land split (keeps 1-of splashes)
  const mb = inferManabase(deckIdx, lands || 17);
  const lh = CO.filter((c) => mb[c] > 0)
    .map((c) => `<span class="lrow"><span class="pip c${c}"></span><b>${mb[c]}</b> ${BASIC_LAND[c]}</span>`).join("")
    || `${lands || 0} colorless`;
  return `<div class="decklands"><b>Lands (${lands || 0}):</b> ${lh}</div>`;
}

// `actualSpells` (optional): nonland indices WITH multiples = the deck the player actually ran. When given,
// a toggle lets you compare it to the model's rebuild; we default to showing the actual played deck.
function openDeckModal(pool, label, actualSpells) {
  S._deckModal = { pool, label, actualSpells: actualSpells && actualSpells.length ? actualSpells : null };
  renderDeckModal(S._deckModal.actualSpells ? "actual" : "model");
  $("deckModal").hidden = false;
}
function renderDeckModal(mode) {
  const st = S._deckModal; if (!st) return;
  const fixed = mode === "actual" ? st.actualSpells : null;
  const r = rateDeck(st.pool, fixed);
  const bar = (x) => `<span class="dbar"><span style="width:${Math.round(x * 100)}%"></span></span>`;
  const pc = Object.entries(r.perTurn).map(([t, p]) => `T${t}:${p.toFixed(2)}`).join("  ");
  const toggle = st.actualSpells                            // only real decks carry an as-run list
    ? `<div class="dtoggle"><button class="${mode === "actual" ? "on" : ""}" onclick="renderDeckModal('actual')">actual played</button>` +
      `<button class="${mode === "model" ? "on" : ""}" onclick="renderDeckModal('model')">model rebuild</button></div>`
    : "";
  $("deckModalBody").innerHTML =
    `<div class="deckhead"><h2>Deck doctor${st.label ? " — " + st.label : ""} · grade ${r.grade} <small>(${r.overall.toFixed(2)})</small></h2>` +
    toggle +
    `<div class="dmeta">${r.deck.length} spells + ${r.lands} lands · ${r.colors.join("/") || "—"} · avg cmc ${r.avgCmc.toFixed(1)}` +
    `${st.actualSpells ? ` · <em>${mode === "actual" ? "the deck as played" : "rebuilt from the pool"}</em>` : ""}</div>` +
    `<div class="dscores"><span>power ${bar(r.powerPct)} ${Math.round(r.powerPct * 100)}th</span>` +
    `<span>castability ${bar(r.cast)} ${r.cast.toFixed(2)}</span><span>coherence ${bar(r.coherence)} ${r.coherence.toFixed(2)}</span></div>` +
    deckLandsHTML(r.deck, r.lands) +
    `<div class="dturns">on-curve by turn: ${pc}</div>` +
    `<ul class="dadvice">${r.advice.map((a) => `<li>${a}</li>`).join("")}</ul></div>` +
    curveColumnsHTML(r.deck);
}
function closeDeckModal() { const m = $("deckModal"); if (m) m.hidden = true; }

function resolveDeckNames(names) {                // card names -> indices (with multiples)
  const byName = {}; S.cards.forEach((c) => (byName[c.name.toLowerCase()] = c.i));
  return names.map((n) => byName[(n || "").toLowerCase()]).filter((i) => i != null);
}

function initDoctor() {
  const sel = $("deckPick"); if (!sel) return;
  const opts = ['<option value="">— pick a deck → full-screen view —</option>'];
  if (S.seats && S.seats[HUMAN] && S.seats[HUMAN].pool.length) opts.push('<option value="__yours">Your drafted deck</option>');
  (S.sampleDecks || []).forEach((d, k) => opts.push(`<option value="${k}">${d.label} (${d.pool.length} cards)</option>`));
  sel.innerHTML = opts.join("");
  sel.onchange = () => {
    const v = sel.value;
    if (v === "") return;
    if (v === "__yours") {
      openDeckModal(S.seats[HUMAN].pool, "your deck");        // no as-run list for an in-progress draft
    } else {
      const d = S.sampleDecks[+v];
      const actual = d.deck ? resolveDeckNames(d.deck) : null; // the spells the player actually ran (w/ multiples)
      openDeckModal(resolveDeckNames(d.pool), d.label, actual);
    }
    sel.value = "";                                // reset so the same deck can be re-opened
  };
  const m = $("deckModal");
  if (m && !m._wired) {                            // wire the modal close once
    m._wired = true;
    $("deckClose").onclick = closeDeckModal;
    m.onclick = (e) => { if (e.target === m) closeDeckModal(); };
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDeckModal(); });
  }
}
