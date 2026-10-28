"""Builds notebooks/value_model_comfort_1028.ipynb; then run
    jupyter nbconvert --to notebook --execute --inplace notebooks/value_model_comfort_1028.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Formulas: the value function, the current model, comfort

- Position $(a, b)$: the side to move has $a$ own words left, the other side $b$.
- Incumbent: `assoc_profile_listener`, the fixed opponent in every win rate.
- Comfort arena: 100 held-out boards × 2 seatings; gpt-oss-120b guesses in two calls
  (the "guess" set $G$ without the number, then a ranking of $G$ with the number; stops at
  a wrong word, after $k$ picks, or when $G$ runs out; never takes $k+1$).
- Data: `cache/win_value_margin.npz`, `cache/report/comfort_frontier_holdout_v1_gptoss_subset_guess.json`.
  No API calls.
""")

code(r"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("..").resolve()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "data")]
from codenames.win_value import WinValue

D = np.load(ROOT / "cache" / "win_value_margin.npz")
n, w, prior, V = D["n"], D["wins"], D["prior"], D["V"]
val = WinValue(ROOT / "cache" / "win_value_margin.npz")
idx = pd.Index(range(1, 10), name="a \\ b")

def grid(T, fmt="{:.3f}"):
    df = pd.DataFrame(T[1:, 1:], index=idx, columns=range(1, 10))
    return df.style.format(lambda x: "–" if pd.isna(x) else fmt.format(x))
""")

# ---------------------------------------------------------------- 1. V
md(r"""
## 1. The value function

$$V(a, b) = P(\text{side to move wins} \mid a, b)$$

From 10,266 incumbent-vs-incumbent games with gpt-oss guessing. One data point per game per
turn-start position: $n_{ab}$ data points, $w_{ab}$ won.
""")

md("Data points per cell, $n_{ab}$:")

code(r"""
print(f"{int(n.sum()):,} data points")
grid(n.astype(float), "{:,.0f}")
""")

md(r"""
**1. Raw rate**
$$\hat p_{ab} = \frac{w_{ab}}{n_{ab}}$$
""")

code(r"""
grid(np.where(n > 0, w / np.maximum(n, 1), np.nan))
""")

md(r"""
**2. Logistic fit**
$$\operatorname{logit} \pi_{ab} = \theta^\top x_{ab}, \qquad
x_{ab} = \left(1,\ a,\ b,\ a^2,\ b^2,\ ab,\ \tfrac{1}{a},\ \tfrac{1}{b}\right)$$
$$\hat\theta = \arg\min_\theta -\sum_{a, b} \Big[ w_{ab} \log \sigma(\theta^\top x_{ab})
+ (n_{ab} - w_{ab}) \log\big(1 - \sigma(\theta^\top x_{ab})\big) \Big]$$
""")

code(r"""
grid(prior)
""")

md(r"""
**3. Shrinkage toward the fit** ($m = 20$ pseudo-games)
$$\tilde V_{ab} = \frac{w_{ab} + m\, \pi_{ab}}{n_{ab} + m}
= (1 - \alpha_{ab})\, \hat p_{ab} + \alpha_{ab}\, \pi_{ab},
\qquad \alpha_{ab} = \frac{m}{n_{ab} + m}$$

Example cells, each step:
""")

code(r"""
shrunk = (w + 20 * prior) / (n + 20)
rows = [{"(a, b)": f"({a}, {b})", "n": int(n[a, b]), "raw": w[a, b] / n[a, b] if n[a, b] else np.nan,
         "fit π": prior[a, b], "α": 20 / (n[a, b] + 20), "shrunk": shrunk[a, b], "V": V[a, b]}
        for a, b in [(9, 8), (4, 4), (8, 8), (8, 9), (8, 2), (9, 2), (3, 9)]]
pd.DataFrame(rows).set_index("(a, b)").style.format(
    {"raw": "{:.3f}", "fit π": "{:.3f}", "α": "{:.1%}", "shrunk": "{:.3f}", "V": "{:.3f}"})
""")

md(r"""
**4. Monotone projection with a margin** ($c = 0.5$; weights $n_{ab} + m$; Dykstra)
$$V = \Pi_{\text{monotone}}\big(\tilde V - c\,\pi\big) + c\,\pi, \qquad
V(a+1, b) \le V(a, b),\quad V(a, b+1) \ge V(a, b)$$

Final $V$:
""")

code(r"""
grid(V)
""")

md(r"""
**Value of one own word**
$$\text{one\_word}(a, b) = V(b, a) - V(b, a-1), \qquad 1 - V(b, 0) = 1$$
""")

code(r"""
pts = [(9, 8), (8, 9), (6, 6), (5, 4), (4, 4), (4, 2), (5, 2), (4, 1), (2, 2), (1, 4)]
pd.DataFrame({"one_word": [val.one_word(a, b) for a, b in pts]},
             index=pd.Index([f"({a}, {b})" for a, b in pts], name="(a, b)")).style.format("{:.3f}")
""")

md(r"""
**Check: fit weighted by $\sqrt{n_{ab}}$ instead of $n_{ab}$.** Fit error is $|\pi_{ab} - \hat p_{ab}|$.
""")

code(r"""
from build_win_value import logistic_fit, monotone_with_margin

def build(pi):
    T = (w + 20 * pi) / (n + 20)
    T[1:, 1:] = monotone_with_margin(T[1:, 1:].copy(), n[1:, 1:] + 20, pi[1:, 1:], 0.5)
    T[0, :], T[:, 0] = 1.0, 0.0
    return T

assert np.allclose(logistic_fit(n, w), prior) and np.allclose(build(prior)[1:, 1:], V[1:, 1:])
s = np.sqrt(n)
prior_sqrt = logistic_fit(np.where(n > 0, s, 0), np.where(n > 0, w / np.maximum(s, 1), 0))
V_sqrt = build(prior_sqrt)
rate = np.where(n > 0, w / np.maximum(n, 1), np.nan)
has = n[1:, 1:] > 0
nn = n[1:, 1:][has]
d = np.abs(V_sqrt - V)[1:, 1:]
out = {}
for name, pi, T in (("weight n (used)", prior, V), ("weight sqrt(n)", prior_sqrt, V_sqrt)):
    err = np.abs(pi - rate)[1:, 1:][has]
    out[name] = {"fit error, game-weighted": np.average(err, weights=nn),
                 "fit error, cells with 20-200 games": err[(nn >= 20) & (nn < 200)].mean(),
                 "opening one_word": T[8, 9] - T[8, 8],
                 "mean |V - V_used|, game-weighted": (np.abs(T - V)[1:, 1:] * n[1:, 1:]).sum() / n.sum()}
pd.DataFrame(out).T.style.format("{:.4f}")
""")

# ---------------------------------------------------------------- 2. the model
md(r"""
## 2. The current model: `floored_overpromise_listener` (λ = 1, φ = 0.05)

1. Shortlist: Gaussian embedding model, best 200 of ~11,000 clues.
2. STOP listener predicts each pick.
3. $\arg\max$ over clue and $k \le \min(a, 4)$ of $W_k(\varnothing)$.

### STOP listener

$$P(x \mid j) = \frac{e^{s_j(x)}}{\sum_{y \notin F} e^{s_j(y)} + e^{s_j(\text{STOP})}}$$

- $s_j$: LightGBM score; 68 features (the incumbent listener's 67 without $k$, plus $j$ and is-STOP).
- Data: 18,687 generated positions, 250 training words; clues aimed 40% own, 20% opponent,
  10% assassin, 10% neutral, 20% random.
- Labels: gpt-oss "guess" set $S$ (asked without the number), ordered by gpt-oss's ranking.
  Events: pick $j$ → $j$-th word of $S$, $j = 1..|S|$; pick $|S|+1$ → STOP.
- Loss: $L = -\sum_e \log P(t_e \mid j_e)$, early-stopped on validation McFadden $R^2$.
- Validation: $R^2 = 1 - \dfrac{\sum_e -\log P(t_e)}{\sum_e \log |C_e|} = 0.481$
  (picks 1–4: 0.603, 0.430, 0.334, 0.493).

The ranking listener (features and shortlist only), McFadden $R^2$:
""")

code(r"""
acc = pd.read_csv(ROOT / "cache" / "report" / "listener_accuracy.csv")
sets = {"train": "train (in-sample)", "held-out words, generated": "held-out words, gpt-oss",
        "held-out words, generated, Sonnet 5.5": "held-out words, Sonnet 5.5"}
r = acc[(acc.group == "features") & acc.model.isin(["incumbent", "+ association profile"]) & acc.set.isin(sets)]
r = r.assign(set=r.set.map(sets), model=r.model.map({"incumbent": "first listener (Sept.)",
                                                     "+ association profile": "association profile (used)"}))
r.set_index(["set", "model"])[["events", "r2", "pick_1", "pick_2", "pick_3", "pick_4"]].style.format(
    {"r2": "{:.3f}", "pick_1": "{:.3f}", "pick_2": "{:.3f}", "pick_3": "{:.3f}", "pick_4": "{:.3f}"})
""")

md(r"""
### Value of a turn

$\ell = a - |F|$, $j = |F|$:
$$u(\ell, b) = 1 - V(b, \ell), \qquad u(0, b) = 1$$
$$W_k(F) = \begin{cases}
1 & \ell = 0\\[2pt]
u(\ell, b) & j = k\\[2pt]
\displaystyle \sum_{i \in \text{own} \setminus F} P(i \mid j)\, W_k(F \cup \{i\})
+ \sum_{x \in \text{bad}} P(x \mid j)\, u_x(\ell)
+ P(\text{STOP} \mid j)\,\big[u(\ell, b) - c\,(k - j)\big] & \text{otherwise}
\end{cases}$$
$$u_x(\ell) = \begin{cases} u(\ell, b) & \text{neutral} \\ u(\ell, b - 1)\ \ (0 \text{ if } b = 1) & \text{opponent} \\ 0 & \text{assassin} \end{cases}$$

### Overpromise charge (where λ enters)

$$c = \lambda \cdot \max\big(\text{one\_word}(a, b),\ \varphi\big), \qquad \lambda = 1,\ \varphi = 0.05$$
""")

code(r"""
pts = [(9, 8), (8, 9), (6, 6), (4, 4), (4, 2), (5, 2), (4, 1), (2, 2)]
pd.DataFrame([{"(a, b)": f"({a}, {b})", "c, no floor": val.one_word(a, b),
               "c, floor 0.05": max(val.one_word(a, b), 0.05)} for a, b in pts]
             ).set_index("(a, b)").style.format("{:.3f}")
""")

# ---------------------------------------------------------------- 3. comfort
md(r"""
## 3. Comfort

$G_t$: the arena guesser's "guess" set for turn $t$ (any colour), number $k_t$; each position counted once.
$$\text{comfort}_t = \frac{\min(|G_t|, k_t)}{k_t}, \qquad
\text{comfort} = \frac{1}{T} \sum_{t=1}^{T} \text{comfort}_t$$

Win rate: share of 200 games against the incumbent. Intervals: 95% bootstrap over boards.
""")

code(r"""
fr = json.loads((ROOT / "cache" / "report" / "comfort_frontier_holdout_v1_gptoss_subset_guess.json").read_text())
tab = pd.DataFrame(fr["rows"])
tab["win% (95% CI)"] = [f"{x:.1%} [{lo:.1%}, {hi:.1%}]" for x, (lo, hi) in zip(tab.win, tab.win_ci)]
tab["comfort (95% CI)"] = [f"{x:.3f} [{lo:.3f}, {hi:.3f}]" for x, (lo, hi) in zip(tab.comfort, tab.comfort_ci)]
front = set(fr.get("frontier", []))
tab["frontier"] = ["*" if i in front or nm in front else "" for i, nm in zip(tab.id, tab.name)]
tab.sort_values("comfort", ascending=False)[["name", "games", "turns", "win% (95% CI)", "comfort (95% CI)", "frontier"]].set_index("name")
""")

md("Win rate against comfort (red: floored, green: unfloored, orange: incumbent):")

code(r"""
fig, ax = plt.subplots(figsize=(8.5, 5.5))
LABEL = {  # name: (colour, label offset in points, alignment)
    "floored margin V + overpromise 1.0, strict": ("C3", (8, -14), "left"),
    "margin V + overpromise 1.0, strict": ("C2", (-8, 10), "right"),
    "assoc_profile (incumbent)": ("C1", (-8, -14), "right"),
    "win_prob (assoc)": ("C0", (6, 6), "left"),
    "win_prob + overpromise 1.0, strict": ("C0", (-8, 8), "right"),
    "overpromise 0.5, strict": ("C0", (8, 6), "left"),
    "overpromise 1.0, strict": ("C0", (-8, -14), "right"),
}
for _, r in tab.iterrows():
    colour, off, ha = LABEL.get(r["name"], ("0.7", None, None))
    ax.errorbar(r.comfort, r.win, xerr=[[r.comfort - r.comfort_ci[0]], [r.comfort_ci[1] - r.comfort]],
                yerr=[[r.win - r.win_ci[0]], [r.win_ci[1] - r.win]], fmt="o", ms=6, color=colour,
                alpha=0.9 if off else 0.6, capsize=2, lw=0.8, zorder=3 if off else 2)
    if off:
        ax.annotate(r["name"], (r.comfort, r.win), fontsize=8, xytext=off, textcoords="offset points", ha=ha,
                    color=colour)
ax.axhline(0.5, color="grey", lw=0.6, ls="--")
ax.set_xlabel("comfort")
ax.set_ylabel("win rate against the incumbent")
plt.tight_layout()
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/value_model_comfort_1028.ipynb")
print("wrote notebooks/value_model_comfort_1028.ipynb")
