/* Rebuild the listener's 44 features for whichever words are still on the board.
 *
 * A port of codenames/listener_features.py::extract. Seventeen of the features
 * depend on the candidate set, so they cannot be precomputed; this recomputes
 * them from per-(clue, word) values that do not change, which is what lets the
 * page refresh every turn the way the real spymaster does.
 *
 * Column order must match FEATURE_NAMES exactly -- the tree walker indexes by
 * position, so a single transposed column is silently wrong rather than an
 * error. tools/check_port.mjs asserts equality against Python.
 */

const SQRT2PI = Math.sqrt(2 * Math.PI);

/* log Phi(x), stable in both tails.
 *
 * Needed by p_max_sigma2, which sums 25 of these per quadrature point; the
 * far-left tail underflows to zero in any implementation that computes Phi
 * first and takes a log afterwards, so the tail is evaluated in log space.
 * Rational approximation is West's double-precision normal CDF.
 */
export function logNdtr(x){
  const z = Math.abs(x);
  let logTail;                                   // log of the upper tail at z
  if (z < 7.07106781186547){
    let b = 3.52624965998911e-02 * z + 0.700383064443688;
    b = b*z + 6.37396220353165;  b = b*z + 33.912866078383;
    b = b*z + 112.079291497871;  b = b*z + 221.213596169931;
    b = b*z + 220.206867912376;
    let d = 8.83883476483184e-02 * z + 1.75566716318264;
    d = d*z + 16.064177579207;   d = d*z + 86.7807322029461;
    d = d*z + 296.564248779674;  d = d*z + 637.333633378831;
    d = d*z + 793.826512519948;  d = d*z + 440.413735824752;
    logTail = -0.5*z*z + Math.log(b) - Math.log(d);
  } else {
    let b = z + 0.65;
    b = z + 4/b; b = z + 3/b; b = z + 2/b; b = z + 1/b;
    logTail = -0.5*z*z - Math.log(b) - Math.log(SQRT2PI);
  }
  if (x >= 0){
    const t = Math.exp(logTail);
    return t < 1e-12 ? -t - t*t/2 : Math.log1p(-t);   // log(1 - tail)
  }
  return logTail;
}

/* Descending rank in [0,1], NaN-safe and tie-safe.
 * An all-NaN column must stay all-NaN rather than being ranked, and ties take
 * their average rank -- getting either wrong turns the feature into "where is
 * this word in the list", which leaked the answer once in the Python. */
export function ranks(v){
  const n = v.length, out = new Float64Array(n).fill(NaN);
  const fin = [];
  for (let i = 0; i < n; i++) if (Number.isFinite(v[i])) fin.push(i);
  if (!fin.length) return out;
  const order = fin.slice().sort((a,b) => v[b] - v[a] || a - b);
  const r = new Float64Array(order.length);
  for (let i = 0; i < order.length; i++) r[i] = i;
  let i = 0;
  while (i < order.length){
    let j = i;
    while (j + 1 < order.length && v[order[j+1]] === v[order[i]]) j++;
    if (j > i){ const avg = (i + j) / 2; for (let t = i; t <= j; t++) r[t] = avg; }
    i = j + 1;
  }
  const denom = Math.max(1, order.length - 1);
  for (let t = 0; t < order.length; t++) out[order[t]] = r[t] / denom;
  return out;
}

/* P(word i is the pick) under perceived = z + N(0, sigma).
 * Conditioning on the winner's own noise makes the others independent, which
 * collapses an orthant probability to this one-dimensional quadrature. */
export function pIsMax(z, sigma){
  const n = z.length;
  if (n === 0) return new Float64Array(0);
  if (n === 1) return Float64Array.from([1]);
  const U = 49, LO = -6, HI = 6, step = (HI - LO) / (U - 1);
  const logW = new Float64Array(U), u = new Float64Array(U);
  for (let c = 0; c < U; c++){
    u[c] = LO + c * step;
    logW[c] = -0.5*u[c]*u[c] - 0.5*Math.log(2*Math.PI) + Math.log(step);
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++){
    let m = -Infinity;
    const inner = new Float64Array(U);
    for (let c = 0; c < U; c++){
      let s = logW[c];
      for (let j = 0; j < n; j++){
        if (j === i) continue;                    // self term dropped
        s += logNdtr(u[c] + (z[i] - z[j]) / sigma);
      }
      inner[c] = s;
      if (s > m) m = s;
    }
    let acc = 0;
    for (let c = 0; c < U; c++) acc += Math.exp(inner[c] - m);
    out[i] = Math.exp(m + Math.log(acc));
  }
  return out;
}

const nanMax = a => { let m = -Infinity, seen = false;
  for (const v of a) if (Number.isFinite(v)){ seen = true; if (v > m) m = v; }
  return seen ? m : NaN; };
const nanSum = a => { let s = 0; for (const v of a) if (Number.isFinite(v)) s += v; return s; };
const gapTop = a => { const m = nanMax(a);
  return Number.isNaN(m) ? a.map(() => NaN) : a.map(v => v - m); };

export const N_FEATURES = 44;
const PV = 19;   // values stored per (clue, word)

/** board: one entry of inputs.json. clue: index into board.clues.
 *  keep: indices of the words still unrevealed. k: the clue number. */
export function buildFeatures(board, clue, keep, k){
  const n = keep.length, pair = board.pair[clue], W = board.word, coh = board.coh;
  // Each column carries its own scale; a single one loses the small-valued
  // SWOW columns entirely (see export_board_inputs.py).
  const g = (w, v) => { const x = pair[w*PV + v]; return x === null ? NaN : x / board.ps[v]; };
  const col = v => keep.map(w => g(w, v));

  const zg = col(0), zn = col(1), zw = col(2);
  const peak = nanMax(zn);
  const sorted = zn.slice().sort((a,b) => b - a);
  const lead = n > 1 ? sorted[0] - sorted[1] : 0;
  const pmax = pIsMax(Float64Array.from(zn), 2.0);
  const gtg = gapTop(zg), gtn = gapTop(zn), gtw = gapTop(zw);

  // cohesion: mean z of w against the clue's other top candidates
  const topN = keep.map((w,i) => i).sort((a,b) => zn[b] - zn[a] || a - b).slice(0, 6);
  const cohRows = [];
  for (const t of topN){
    const src = keep[t] * board.words.length;
    const row = keep.map(w => { const x = coh[src + w]; return x === null ? NaN : x / board.cs; });
    if (row.some(Number.isFinite)) cohRows.push([row, t]);
  }
  const cohesion = new Array(n).fill(NaN);
  if (cohRows.length){
    for (let i = 0; i < n; i++){
      let s = 0, c = 0;
      for (const [row, t] of cohRows) if (t !== i && Number.isFinite(row[i])){ s += row[i]; c++; }
      if (c) cohesion[i] = s / c;
    }
  }

  const s1 = col(3), s2 = col(4), r1 = col(5), r2 = col(6);
  const tot = nanSum(s2), rtot = nanSum(r2);
  const fwdShare = s2.map(v => tot > 0 ? v / tot : NaN);
  const revShare = r2.map(v => rtot > 0 ? v / rtot : NaN);
  const ent = col(7), entR = ranks(ent);
  const pmiGap = gapTop(col(8));
  const g840 = col(9), g840R = ranks(g840), g840G = gapTop(g840);
  const ft = col(10), ftG = gapTop(ft);
  const wup = col(11), wupR = ranks(wup);
  const lex = [12,13,14,15,16,17,18].map(col);
  const wordv = j => keep.map(w => { const x = W[w*7 + j]; return x === null ? NaN : x / board.ws[j]; });
  const wm = wordv(0), wsd = wordv(1), conc = wordv(2), concSd = wordv(3),
        pk = wordv(4), lf = wordv(5), ns = wordv(6);

  const rows = [];
  for (let i = 0; i < n; i++){
    rows.push(Float64Array.from([
      zg[i], zn[i], gtg[i], gtn[i], gtw[i],
      k, n, peak, lead, pmax[i],
      Math.min(zg[i], zn[i], zw[i]),
      wm[i], wsd[i], cohesion[i], cohesion[i] - zn[i],
      s1[i], s2[i], Number.isNaN(s2[i]) ? 0 : 1,
      r1[i], r2[i], revShare[i], revShare[i] - fwdShare[i],
      entR[i], Number.isNaN(ent[i]) ? 0 : 1,
      pmiGap[i],
      g840[i], g840R[i], g840G[i], ft[i], ftG[i],
      conc[i], concSd[i], pk[i], lf[i], ns[i],
      wup[i], wupR[i],
      lex[0][i], lex[1][i], lex[2][i], lex[3][i], lex[4][i], lex[5][i], lex[6][i],
    ]));
  }
  return rows;
}
