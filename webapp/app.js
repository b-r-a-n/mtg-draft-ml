/* Static draft-pod simulator. The content draft model runs in-browser via onnxruntime-web (WASM):
 * we generate real boosters from a set, deal an 8-seat pod, auto-draft the 7 bot seats with the
 * model (+ per-seat aggressiveness on the deck_value dial), and overlay the model's read on every
 * human pick. Passing/signals emerge from the pod exactly like a real draft. No server. */

const N_SEATS = 8, HUMAN = 0, N_PACKS = 3;
// Synthetic fallback = Play Booster slots (14 cards) if no real-pack library is available.
const SLOTS = { raremythic: 1, uncommon: 3, common: 8, wildcard: 2 };
const COLORS = ["W", "U", "B", "R", "G", "C"];

const S = {                                              // global state
  session: null, meta: null, cards: [], byRarity: {}, qz: [], realPacks: null, playprob: null, MAXP: 45, MAXK: 15,
  seats: [], packs: [], round: 0, pick: 0, dir: 1, humanAgg: 0, botAgg: 3, busy: false,
  stats: null, seen: [],
};

const $ = (id) => document.getElementById(id);
const setStatus = (t) => { $("status").textContent = t; };

// ---- load -------------------------------------------------------------------------------------
async function loadSet(setCode) {
  setStatus(`loading ${setCode}…`);
  const meta = await (await fetch(`data/${setCode}.meta.json`)).json();
  const cards = (await (await fetch(`data/${setCode}.cards.json`)).json()).cards;
  S.realPacks = await fetch(`data/${setCode}.packs.json`).then((r) => r.ok ? r.json() : null).catch(() => null);
  // learned buildability deckbuilder (P(played|pool) as JSON trees); falls back to deck_value sort if absent
  S.playprob = await fetch(`model/${setCode}.playprob.json`).then((r) => r.ok ? r.json() : null).catch(() => null);
  S.meta = meta; S.cards = cards; S.MAXP = meta.max_pool; S.MAXK = meta.max_pack;
  // z-score the dial signal (deck_value) over rated cards, like deploy.Drafter
  const qs = cards.map((c) => c.q).filter((v) => v != null && isFinite(v));
  const mu = qs.reduce((a, b) => a + b, 0) / Math.max(qs.length, 1);
  const sd = Math.sqrt(qs.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(qs.length, 1)) || 1;
  S.qz = cards.map((c) => (c.q == null || !isFinite(c.q)) ? 0 : (c.q - mu) / sd);
  S.byRarity = { rare: [], mythic: [], uncommon: [], common: [] };
  cards.forEach((c) => (S.byRarity[c.rarity] || S.byRarity.common).push(c.i));
  S.session = await ort.InferenceSession.create(`${meta.onnx}`, { executionProviders: ["wasm"] });
  setStatus(`${setCode} ready · model emb${meta.emb_dim}/h${meta.enc_hidden} · ${cards.length} cards` +
    (meta.target_in_train ? "" : " · (held-out: model never trained on this set)"));
}

// ---- booster generation -----------------------------------------------------------------------
function sample(pool, n, used) {
  const out = [], avail = pool.filter((i) => !used.has(i));
  for (let k = 0; k < n && avail.length; k++) {
    const j = Math.floor(Math.random() * avail.length);
    out.push(avail[j]); used.add(avail[j]); avail.splice(j, 1);
  }
  return out;
}
function makeBooster() {
  // Primary: deal a real opened pack from the 17lands draft data (empirically-correct distribution).
  if (S.realPacks && S.realPacks.length) {
    return [...S.realPacks[Math.floor(Math.random() * S.realPacks.length)]];
  }
  // Fallback: synthesize a Play Booster (14 cards) from rarity slots + 2 any-rarity wildcards.
  const used = new Set(), out = [];
  const rm = (Math.random() < 1 / 8 && S.byRarity.mythic.length) ? S.byRarity.mythic : S.byRarity.rare;
  out.push(...sample(rm, SLOTS.raremythic, used));
  out.push(...sample(S.byRarity.uncommon, SLOTS.uncommon, used));
  out.push(...sample(S.byRarity.common, SLOTS.common + 1, used));  // 7 commons (incl. land slot)
  for (let w = 0; w < SLOTS.wildcard; w++) {                       // wildcards: any rarity, weighted
    const r = Math.random();
    const pool = r < 0.5 ? S.byRarity.common : r < 0.82 ? S.byRarity.uncommon
      : r < 0.96 ? S.byRarity.rare : (S.byRarity.mythic.length ? S.byRarity.mythic : S.byRarity.rare);
    out.push(...sample(pool.length ? pool : S.byRarity.common, 1, used));
  }
  return out;
}

// ---- inference (batched over seats) -----------------------------------------------------------
async function inferAll() {
  const B = N_SEATS, MAXP = S.MAXP, MAXK = S.MAXK;
  const pool = new BigInt64Array(B * MAXP), poolM = new Uint8Array(B * MAXP);
  const pack = new BigInt64Array(B * MAXK), packM = new Uint8Array(B * MAXK);
  for (let s = 0; s < B; s++) {
    const pl = S.seats[s].pool, pk = S.packs[s];
    for (let j = 0; j < pl.length && j < MAXP; j++) { pool[s * MAXP + j] = BigInt(pl[j]); poolM[s * MAXP + j] = 1; }
    for (let j = 0; j < pk.length && j < MAXK; j++) { pack[s * MAXK + j] = BigInt(pk[j]); packM[s * MAXK + j] = 1; }
  }
  const t = (d, a, dim) => new ort.Tensor(d, a, dim);
  const out = await S.session.run({
    pool: t("int64", pool, [B, MAXP]), pool_mask: t("bool", poolM, [B, MAXP]),
    pack: t("int64", pack, [B, MAXK]), pack_mask: t("bool", packM, [B, MAXK]),
  });
  const L = out.logits.data;                              // Float32 [B, MAXK]
  return S.seats.map((_, s) => Array.from(L.slice(s * MAXK, s * MAXK + S.packs[s].length)));
}

// effective score = model logit + aggressiveness * z(deck_value)
const eff = (logits, pack, agg) => logits.map((l, k) => l + agg * S.qz[pack[k]]);
const argmax = (a) => a.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
const softmax = (a) => { const m = Math.max(...a), e = a.map((x) => Math.exp(x - m)), s = e.reduce((p, q) => p + q, 0); return e.map((x) => x / s); };

// ---- draft loop ---------------------------------------------------------------------------------
async function startDraft() {
  S.humanAgg = +$("humanAgg").value; S.botAgg = +$("botAgg").value;
  S.seats = Array.from({ length: N_SEATS }, (_, i) => ({
    pool: [], isHuman: i === HUMAN,
    agg: i === HUMAN ? 0 : Math.random() * S.botAgg,      // each bot its own aggressiveness
  }));
  S.stats = { matches: 0, picks: 0, regret: 0, dvSum: 0, dvN: 0 };
  S.seen = []; S.round = 0;
  $("summaryPanel").hidden = true;
  renderSeats();
  await newRound();
}

async function newRound() {
  S.packs = Array.from({ length: N_SEATS }, makeBooster);
  S.dir = (S.round % 2 === 0) ? 1 : -1;                   // pack 1 left, pack 2 right, pack 3 left
  S.pick = 0;
  await nextPick();
}

async function nextPick() {
  if (S.packs[HUMAN].length === 0) {                      // round over
    S.round++;
    if (S.round >= N_PACKS) return finishDraft();
    return newRound();
  }
  S.busy = true;
  $("packTitle").textContent = `Pack ${S.round + 1}, Pick ${S.pick + 1}`;
  const logits = await inferAll();
  // bots commit their picks now (simultaneous with the human deliberating)
  for (let s = 0; s < N_SEATS; s++) {
    if (s === HUMAN) continue;
    const sc = eff(logits[s], S.packs[s], S.seats[s].agg);
    const choice = S.packs[s][argmax(sc)];
    S.seats[s].pool.push(choice);
    S.packs[s] = S.packs[s].filter((c) => c !== choice);
  }
  renderHumanPack(logits[HUMAN]);                          // overlay + wait for click
  renderSeats();
  S.busy = false;
}

function humanPick(cardIdx, modelTop, packBefore) {
  if (S.busy) return;
  // assessment stats vs the model's recommendation (at the human's overlay aggressiveness)
  const dv = S.cards[cardIdx].deck_value;
  const dvs = packBefore.map((c) => S.cards[c].deck_value).filter((v) => v != null);
  S.stats.picks++;
  if (cardIdx === modelTop) S.stats.matches++;
  if (dv != null) { S.stats.dvSum += dv; S.stats.dvN++; if (dvs.length) S.stats.regret += Math.max(...dvs) - dv; }
  S.seats[HUMAN].pool.push(cardIdx);
  S.packs[HUMAN] = S.packs[HUMAN].filter((c) => c !== cardIdx);
  // pass: rotate the in-flight packs around the table
  S.packs = S.dir === 1 ? [...S.packs.slice(1), S.packs[0]] : [S.packs[S.packs.length - 1], ...S.packs.slice(0, -1)];
  S.seen.push(...S.packs[HUMAN].map((c) => S.cards[c].ci)); // colors of what's being passed to you
  S.pick++;
  renderPool();
  nextPick();
}

// Curve-based land count: avg deck_value can't choose lands (more lands trivially raises the avg by
// cutting weak spells), so the land count is a curve/consistency call — low curve -> 16, high -> 18.
function recommendLands(cmcs) {
  const avg = cmcs.length ? cmcs.reduce((a, b) => a + b, 0) / cmcs.length : 3;
  return { lands: avg < 2.6 ? 16 : avg > 3.3 ? 18 : 17, avgCmc: avg };
}

// ---- buildability deckbuilder: P(played|pool) as JSON trees (mirrors eval/play_prob.py) ----------
const PP_COLORS = ["W", "U", "B", "R", "G"];
function ppMargin(feat, pp) {                       // tree-walk -> raw margin (monotone in P(played); ranking only)
  let s = pp.base;
  for (const t of pp.trees) {
    let i = 0;
    while (!t.leaf[i]) {
      const x = feat[t.f[i]];
      i = Number.isNaN(x) ? (t.ml[i] ? t.l[i] : t.r[i]) : (x <= t.thr[i] ? t.l[i] : t.r[i]);
    }
    s += t.val[i];
  }
  return s;
}
function ppPoolFeats(pool) {                         // [12] = poolcolor[5] + pool cmc-bucket[1..6] + size
  const col = [0, 0, 0, 0, 0], cmc = [0, 0, 0, 0, 0, 0]; let size = 0;
  for (const idx of pool) {
    const c = S.cards[idx], ci = c.ci || "";
    for (let k = 0; k < 5; k++) if (ci.includes(PP_COLORS[k])) col[k] += 1;
    const b = Math.min(6, Math.max(1, Math.round(c.cmc || 0)));   // cmc rounded, clipped 1..6
    cmc[b - 1] += 1; size += 1;
  }
  return [...col, ...cmc, size];
}
function ppCardFeat(idx) {                           // [9] = [cmc, deck_value, W,U,B,R,G, creature, land]
  const c = S.cards[idx], ci = c.ci || "";
  const dv = (c.deck_value == null || !isFinite(c.deck_value)) ? NaN : c.deck_value;
  return [c.cmc || 0, dv, ...PP_COLORS.map((k) => ci.includes(k) ? 1 : 0),
    c.t === "creature" ? 1 : 0, c.t === "land" ? 1 : 0];
}
// nonland pool cards ranked by P(played|pool) — the learned buildability deck (mirrors deck_from_play_model)
function buildSpellsPlayprob(pool) {
  const cand = [...new Set(pool)].filter((i) => S.cards[i].t !== "land");   // unique nonland (dict.fromkeys)
  if (!cand.length) return [];
  const pf = ppPoolFeats(pool);                                            // over the FULL pool (dupes count)
  return cand.map((i, k) => ({ i, k, m: ppMargin([...ppCardFeat(i), ...pf], S.playprob) }))
    .sort((a, b) => (b.m - a.m) || (a.k - b.k))                            // desc; stable tie-break by pool order
    .map((x) => x.i);
}

function seatSummary(pool) {
  // build the deck from the best nonland spells: P(played|pool) if available (learned buildability,
  // ~2-color), else the context-free deck_value sort. Land count from the chosen spells' curve.
  const nonland = S.playprob
    ? buildSpellsPlayprob(pool)
    : pool.filter((i) => S.cards[i].t !== "land" && S.cards[i].deck_value != null)
      .sort((a, b) => S.cards[b].deck_value - S.cards[a].deck_value);
  const { lands, avgCmc } = recommendLands(nonland.slice(0, 23).map((i) => S.cards[i].cmc || 0));
  const play = nonland.slice(0, 40 - lands);
  const avg = play.length ? play.reduce((a, i) => a + (S.cards[i].deck_value || 0), 0) / play.length : 0;
  // colors off the BUILT deck (not the whole pool) — the deck the model would actually register/run
  const cc = {}; play.forEach((i) => (S.cards[i].ci || "").split("").forEach((c) => c !== "C" && (cc[c] = (cc[c] || 0) + 1)));
  const colors = Object.entries(cc).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([c]) => c);
  return { avg, colors, lands, avgCmc, nSpells: play.length };
}

// color x type matrix for a pool: {color: {creature, spell, land}}
function deckBreakdown(pool) {
  const m = {}; const cats = ["creature", "spell", "land"];
  pool.forEach((i) => {
    const c = S.cards[i], cols = (c.ci || "C").split("").filter((x) => x) ;
    (cols.length ? cols : ["C"]).forEach((col) => {
      m[col] = m[col] || { creature: 0, spell: 0, land: 0 };
      m[col][c.t] += 1;
    });
  });
  return { m, cats };
}
const colorPips = (cs) => (cs.length ? cs : ["C"]).map((c) => `<span class="pip c${c}"></span>`).join("");

function finishDraft() {
  $("pack").innerHTML = ""; $("packTitle").textContent = "Draft complete";
  const matchPct = S.stats.picks ? (100 * S.stats.matches / S.stats.picks) : 0;
  const avgRegret = S.stats.dvN ? S.stats.regret / S.stats.dvN : 0;

  // summarize + rank every seat's deck
  const rows = S.seats.map((s, i) => ({ i, you: s.isHuman, agg: s.agg, ...seatSummary(s.pool) }));
  rows.sort((a, b) => b.avg - a.avg);
  const youRank = rows.findIndex((r) => r.you) + 1;
  const you = rows.find((r) => r.you);

  const bd = deckBreakdown(S.seats[HUMAN].pool);
  const usedColors = Object.keys(bd.m).filter((c) => c !== "C" && (bd.m[c].creature + bd.m[c].spell + bd.m[c].land))
    .sort((a, b) => (bd.m[b].creature + bd.m[b].spell) - (bd.m[a].creature + bd.m[a].spell));
  const cols = [...usedColors, ...(bd.m.C ? ["C"] : [])];

  $("summaryPanel").hidden = false;
  $("summary").innerHTML =
    `<div class="row"><span>Your deck (best ${you.nSpells} spells + ${you.lands} lands)</span><b>${you.avg.toFixed(3)}</b></div>` +
    `<div class="row"><span>Suggested lands <small>(avg CMC ${you.avgCmc.toFixed(1)})</small></span><b>${you.lands}</b></div>` +
    `<div class="row"><span>Pod finish</span><b>#${youRank} of ${N_SEATS}</b></div>` +
    `<div class="row"><span>You matched the model</span><b>${matchPct.toFixed(0)}%</b> <small>(${S.stats.matches}/${S.stats.picks})</small></div>` +
    `<div class="row"><span>Avg value left in pack</span><b>${avgRegret.toFixed(3)}</b></div>` +
    `<h3>Your deck — type × color</h3>` +
    `<table class="breakdown"><tr><th></th>${cols.map((c) => `<th><span class="pip c${c}"></span></th>`).join("")}<th>Σ</th></tr>` +
    bd.cats.map((cat) => `<tr><td>${cat[0].toUpperCase() + cat.slice(1)}s</td>` +
      cols.map((c) => `<td>${bd.m[c][cat] || ""}</td>`).join("") +
      `<td class="mono">${cols.reduce((a, c) => a + (bd.m[c][cat] || 0), 0)}</td></tr>`).join("") +
    `</table>` +
    `<h3>Pod comparison <small>(best-N spells by deck_value)</small></h3>` +
    `<div class="podtable">` + rows.map((r, rank) =>
      `<div class="prow ${r.you ? "you" : ""}"><span class="rk">${rank + 1}</span>` +
      `<span class="nm">Seat ${r.i + 1}${r.you ? " — you" : ` <small>agg ${r.agg.toFixed(1)}</small>`}</span>` +
      `<span class="pips">${colorPips(r.colors)}</span>` +
      `<span class="lc mono">${r.nSpells}+${r.lands}</span>` +
      `<span class="dv mono">${r.avg.toFixed(3)}</span></div>`).join("") + `</div>`;
  renderPool();
}

// ---- rendering ----------------------------------------------------------------------------------
function renderHumanPack(logits) {
  const pack = S.packs[HUMAN];
  const scores = eff(logits, pack, S.humanAgg), probs = softmax(scores), top = argmax(scores);
  const modelTopIdx = pack[top];
  const order = pack.map((_, k) => k).sort((a, b) => probs[b] - probs[a]);
  const el = $("pack"); el.innerHTML = "";
  const before = [...pack];
  order.forEach((k) => {
    const ci = pack[k], c = S.cards[ci], dv = c.deck_value;
    const div = document.createElement("div");
    div.className = "card" + (k === top ? " modelpick" : "");
    div.innerHTML =
      `<span class="badge r-${c.rarity}">${(probs[k] * 100).toFixed(0)}%${k === top ? " ★" : ""}</span>` +
      (c.img ? `<img loading="lazy" src="${c.img}" alt="${c.name}">` : `<div style="padding:8px">${c.name}</div>`) +
      `<div class="ov"><div class="bar"><i style="width:${(probs[k] * 100).toFixed(0)}%"></i></div>` +
      `<div class="dv"><span>${c.name.length > 18 ? c.name.slice(0, 17) + "…" : c.name}</span>` +
      `<span class="${dv == null ? '' : dv >= 0 ? 'pos' : 'neg'}">${dv == null ? "·" : dv.toFixed(2)}</span></div></div>`;
    div.onclick = () => humanPick(ci, modelTopIdx, before);
    div.onmousemove = (e) => showTip(e, c); div.onmouseleave = hideTip;
    el.appendChild(div);
  });
  $("signals").textContent = `Open (passed to you): ${topColors(S.seen.slice(-45))}`;
}

function topColors(arr) {
  const c = {}; arr.forEach((x) => x.split("").forEach((ch) => COLORS.includes(ch) && (c[ch] = (c[ch] || 0) + 1)));
  const tot = Object.values(c).reduce((a, b) => a + b, 0) || 1;
  return Object.entries(c).filter(([k]) => k !== "C").sort((a, b) => b[1] - a[1]).slice(0, 3)
    .map(([k, v]) => `${k} ${(100 * v / tot).toFixed(0)}%`).join("  ") || "—";
}

function renderPool() {
  const pool = S.seats[HUMAN].pool;
  $("poolCount").textContent = pool.length;
  const counts = {}; pool.forEach((i) => { const ci = S.cards[i].ci || "C"; ci.split("").forEach((c) => counts[c] = (counts[c] || 0) + 1); });
  const tot = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  $("poolColors").innerHTML = COLORS.map((c) => counts[c] ? `<span class="c${c}" style="width:${100 * counts[c] / tot}%"></span>` : "").join("");
  const byCmc = [...pool].sort((a, b) => (S.cards[a].cmc || 0) - (S.cards[b].cmc || 0));
  $("pool").innerHTML = byCmc.map((i) => {
    const c = S.cards[i];
    return `<div class="pl"><span>${c.name}</span><small>${c.ci} · ${c.deck_value == null ? "·" : c.deck_value.toFixed(2)}</small></div>`;
  }).join("");
}

function renderSeats() {
  $("seats").innerHTML = S.seats.map((s, i) =>
    `<div class="seat ${s.isHuman ? "you" : ""}"><span>Seat ${i + 1}${s.isHuman ? " — you" : ""}</span>` +
    `<span class="agg">${s.isHuman ? `pool ${s.pool.length}` : `agg ${s.agg.toFixed(1)} · ${s.pool.length}`}</span></div>`).join("");
}

function showTip(e, c) {
  if (!c.img) return; const t = $("cardtip");
  t.innerHTML = `<img src="${c.img}">`; t.hidden = false;
  const x = Math.min(e.clientX + 16, innerWidth - 256);
  t.style.left = x + "px"; t.style.top = Math.min(e.clientY + 16, innerHeight - 340) + "px";
}
const hideTip = () => ($("cardtip").hidden = true);

// ---- boot ---------------------------------------------------------------------------------------
async function boot() {
  const sets = await (await fetch("data/sets.json")).json();
  $("setSel").innerHTML = sets.map((s) => `<option>${s}</option>`).join("");
  $("humanAgg").oninput = (e) => $("humanAggVal").textContent = e.target.value;
  $("botAgg").oninput = (e) => $("botAggVal").textContent = e.target.value;
  $("setSel").onchange = async (e) => { await loadSet(e.target.value); };
  $("newDraft").onclick = async () => { if (!S.session) await loadSet($("setSel").value); startDraft(); };
  if (sets.length) await loadSet(sets[0]);
}
boot();
