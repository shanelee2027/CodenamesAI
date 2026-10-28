"""Builds notebooks/value_model_comfort_1028.ipynb; then run
    jupyter nbconvert --to notebook --execute --inplace notebooks/value_model_comfort_1028.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Formulas: the value function, the current model, comfort

| symbol | |
|---|---|
| $(a, b)$ | own words left, other side's words left (side to move) |
| $k$ | the clue's number |
| $F$, $j = \lvert F\rvert$ | own words found so far this turn, picks made |
| $\pi_{ab}$ | the logistic fit (a table), not 3.14 |
| $\Pi$ | projection onto monotone tables |
| incumbent | `assoc_profile_listener` |
| arena | 100 held-out boards × 2 seatings, gpt-oss-120b guessing: $G$ = "guess" set (no number) → rank $G$ (with $k$) → stop at a wrong word, $k$ picks, or $G$ exhausted |
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

$$n_{ab} = \#\{(\text{game}, \text{turn start at } (a, b))\}, \qquad
w_{ab} = \#\{\ldots \text{won by the side to move}\}, \qquad 10{,}266 \text{ incumbent games}$$

$n_{ab}$:
""")

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
""")

code(r"""
shrunk = (w + 20 * prior) / (n + 20)
rows = [{"(a, b)": f"({a}, {b})", "n": int(n[a, b]), "raw": w[a, b] / n[a, b] if n[a, b] else np.nan,
         "fit π": prior[a, b], "α": 20 / (n[a, b] + 20), "shrunk Ṽ": shrunk[a, b]}
        for a, b in [(9, 8), (4, 4), (8, 8), (8, 9), (7, 2), (8, 2), (9, 2), (3, 9)]]
pd.DataFrame(rows).set_index("(a, b)").style.format(
    {"raw": "{:.3f}", "fit π": "{:.3f}", "α": "{:.1%}", "shrunk Ṽ": "{:.3f}"})
""")

md(r"""
**4. Monotone projection with a margin** ($c = 0.5$, $\omega_{ab} = n_{ab} + m$)

Plain projection (2-D isotonic regression):
$$\Pi(X) = \arg\min_{V} \sum_{a,b} \omega_{ab}\,\big(V_{ab} - X_{ab}\big)^2
\quad \text{s.t.} \quad V_{a+1,\,b} \le V_{ab}, \quad V_{a,\,b+1} \ge V_{ab}$$

With the margin:
$$V = \Pi\big(\tilde V - c\,\pi\big) + c\,\pi$$

$$\Longleftrightarrow\quad
\boxed{\;V = \arg\min_{V} \sum_{a,b} \omega_{ab}\big(V_{ab} - \tilde V_{ab}\big)^2
\quad \text{s.t.}\quad
\begin{aligned}
V_{a,b+1} - V_{ab} &\ge c\,\big(\pi_{a,b+1} - \pi_{ab}\big)\\
V_{ab} - V_{a+1,b} &\ge c\,\big(\pi_{ab} - \pi_{a+1,b}\big)
\end{aligned}\;}$$

$c = 0$: ties allowed (old V, `win_value.npz`). $c = 1$: leaves $[0, 1]$. Computed by Dykstra (alternating row / column isotonic regression).

Same cells, step 3 → step 4. $(7,2), (8,2), (9,2)$ pooled: $\tfrac{191 \cdot 0.018 + 51 \cdot 0.006 + 33 \cdot 0.158}{275} = 0.033$, then spread by the margin.
""")

code(r"""
pd.DataFrame([{"(a, b)": f"({a}, {b})", "ω": n[a, b] + 20, "π": prior[a, b], "shrunk Ṽ": shrunk[a, b], "V": V[a, b],
               "V − Ṽ": V[a, b] - shrunk[a, b]}
              for a, b in [(9, 8), (4, 4), (8, 8), (8, 9), (7, 2), (8, 2), (9, 2), (3, 9)]]).set_index("(a, b)").style.format(
    {"ω": "{:.0f}", "π": "{:.4f}", "shrunk Ṽ": "{:.4f}", "V": "{:.4f}", "V − Ṽ": "{:+.4f}"})
""")

code(r"""
print(f"fit step {prior[8, 9] - prior[8, 8]:.4f}   required ≥ {0.5 * (prior[8, 9] - prior[8, 8]):.4f}   "
      f"V step {V[8, 9] - V[8, 8]:.4f}   Ṽ step {shrunk[8, 9] - shrunk[8, 8]:.4f}")
""")

md(r"""
$V$:
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
**Check:** fit weights $\sqrt{n_{ab}}$ vs $n_{ab}$; error $= |\pi_{ab} - \hat p_{ab}|$.
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

$$(\text{clue}^*, k^*) = \arg\max_{\text{clue} \in \text{top-200},\ k \le \min(a, 4)} W_k(\varnothing)$$

(top-200: Gaussian embedding score over ~11,000 clues)

### STOP listener

$$P(x \mid j) = \frac{e^{s_j(x)}}{\sum_{y \notin F} e^{s_j(y)} + e^{s_j(\text{STOP})}},
\qquad s_j(x) = \text{GBT}\big(\phi_{\text{assoc}}(\text{clue}, x) \setminus k,\ j,\ \mathbb{1}[x = \text{STOP}]\big) \quad (68 \text{ features})$$

Data: 18,687 positions, 250 training words,
$$\text{clue aimed at} \sim \{\text{own } 0.4,\ \text{opp } 0.2,\ \text{assassin } 0.1,\ \text{neutral } 0.1,\ \text{random } 0.2\}$$

Labels: $S = (s_1, \dots, s_{|S|})$, gpt-oss "guess" set (no number), ordered by gpt-oss's ranking:
$$t_j = s_j \ \ (j = 1, \dots, |S|), \qquad t_{|S|+1} = \text{STOP}$$

$$L = -\sum_e \log P(t_e \mid j_e), \qquad
R^2 = 1 - \frac{\sum_e -\log P(t_e)}{\sum_e \log |C_e|} = 0.481 \quad (\text{picks } 1\text{–}4:\ 0.603,\ 0.430,\ 0.334,\ 0.493)$$

Ranking listener $R^2$ (features and shortlist only):
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

$$\ell = a - |F|, \qquad u(\ell, b) = 1 - V(b, \ell), \qquad u(0, b) = 1$$
$$W_k(F) = \begin{cases}
1 & \ell = 0\\[2pt]
u(\ell, b) & j = k\\[2pt]
\displaystyle \sum_{i \in \text{own} \setminus F} P(i \mid j)\, W_k(F \cup \{i\})
+ \sum_{x \in \text{bad}} P(x \mid j)\, u_x(\ell)
+ P(\text{STOP} \mid j)\,\big[u(\ell, b) - c\,(k - j)\big] & \text{otherwise}
\end{cases}$$
$$u_x(\ell) = \begin{cases} u(\ell, b) & \text{neutral} \\ u(\ell, b - 1)\ \ (0 \text{ if } b = 1) & \text{opponent} \\ 0 & \text{assassin} \end{cases}$$

### Overpromise charge

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

$$\text{comfort}_t = \frac{\min(|G_t|, k_t)}{k_t}, \qquad
\text{comfort} = \frac{1}{T} \sum_{t=1}^{T} \text{comfort}_t$$

$G_t$: "guess" set (any colour); distinct positions only. Win rate over 200 games vs the incumbent; 95% CIs bootstrap over boards.
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

md("Red: floored. Green: unfloored. Orange: incumbent.")

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
