/* Evaluate the exported LightGBM model.
 *
 * Mirrors LightGBM's numerical decision exactly, including the missing-value
 * rules, which matter more here than they usually would: 41% of clues have no
 * SWOW row and 71% no entity row, so NaN is the common path, not an edge case.
 *
 *   isnan(v) && !missingIsNaN  ->  v = 0, then compare
 *   missingIsNaN && isnan(v)   ->  take default_left
 *   otherwise                  ->  v <= threshold ? left : right
 *
 * Flags: bit 0 = default_left, bit 1 = missing_type is NaN.
 */
export function makePredictor(model){
  const trees = model.trees.map(t => ({
    r: t.r,
    f: Int32Array.from(t.f), t: Float64Array.from(t.t),
    l: Int32Array.from(t.l), g: Int32Array.from(t.g),
    m: Uint8Array.from(t.m), v: Float64Array.from(t.v),
  }));
  return function predict(x){          // x: Float64Array of features, one row
    let sum = 0;
    for (const tr of trees){
      let n = tr.r;
      while (n >= 0){
        let v = x[tr.f[n]];
        const flag = tr.m[n], defLeft = (flag & 1) !== 0, missNaN = (flag & 2) !== 0;
        let goLeft;
        if (Number.isNaN(v)){
          if (missNaN) goLeft = defLeft;
          else goLeft = (0 <= tr.t[n]);     // NaN coerced to 0, then compared
        } else {
          goLeft = v <= tr.t[n];
        }
        n = goLeft ? tr.l[n] : tr.g[n];
      }
      sum += tr.v[~n];
    }
    return sum;
  };
}
