/* Expected reward of a clue, per docs/clue-selection-learned.tex.
 *
 * Words race with rate exp(score) and the turn ends at the first non-team word.
 * Substituting u = exp(-Lambda t) maps the integral to [0,1], so the CELLS
 * midpoints each carry equal probability mass. Shared by the page and by the
 * port's end-to-end check so there is one implementation, not two.
 */
export const COST = {n: 0.2, b: 1.0, x: 10.0};   // neutral, opponent, assassin
export const CELLS = 48;

export function evaluate(scores, ownIdx, badIdx, badCost, kmax){
  let mx = -Infinity;
  for (const i of ownIdx) if (scores[i] > mx) mx = scores[i];
  for (const i of badIdx) if (scores[i] > mx) mx = scores[i];
  const lo = ownIdx.map(i => Math.exp(scores[i] - mx));
  let L = 0, cb = 0;
  for (let t = 0; t < badIdx.length; t++){
    const l = Math.exp(scores[badIdx[t]] - mx);
    L += l; cb += l * badCost[t];
  }
  if (L <= 0) L = 1e-300;
  cb /= L;
  const ratio = lo.map(l => l / L);
  const gain = new Float64Array(kmax), pen = new Float64Array(kmax);
  const q = new Float64Array(kmax + 1);
  for (let c = 0; c < CELLS; c++){
    const u = (c + 0.5) / CELLS;
    q.fill(0); q[0] = 1;
    for (const r of ratio){
      const p = 1 - Math.pow(u, r);
      for (let j = kmax; j >= 1; j--) q[j] = q[j]*(1-p) + q[j-1]*p;
      q[0] *= (1 - p);
    }
    let cum = 0;
    for (let k = 0; k < kmax; k++){
      cum += q[k];
      const tail = 1 - cum;
      for (let kk = k; kk < kmax; kk++) gain[kk] += tail / CELLS;
      pen[k] += cum / CELLS;
    }
  }
  let bk = 0, bv = -Infinity;
  for (let k = 0; k < kmax; k++){ const v = gain[k] - cb*pen[k]; if (v > bv){ bv = v; bk = k; } }
  return {k: bk + 1, value: bv};
}
