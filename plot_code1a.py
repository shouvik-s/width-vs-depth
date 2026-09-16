"""Shallow-vs-deep figures for code1a.py."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch, code1a

torch.set_num_threads(1)
NS = range(2, 9)
# min sufficient width measured by run_code1a.py (tol=1e-3, best of 5 seeds)
SHALLOW = {2: (4, 13), 3: (16, 49)}          # n -> (w, params); absent => w > 128
DEEP = {n: (2, 6 * n + 1) for n in range(2, 6)}   # w=2 fixed; fails n>=6
DEEP_MSE = {2: 3.416e-11, 3: 5.549e-10, 4: 2.776e-09, 5: 2.281e-07,
            6: 5.037e-02, 7: 8.337e-02, 8: 8.338e-02}
SHAL_MSE = {2: 1.328e-06, 3: 2.830e-05, 4: 1.942e-02, 5: 7.156e-02,
            6: 8.088e-02, 7: 8.391e-02, 8: 8.403e-02}

# ---------------- Figure 1: cost vs depth ----------------
fig, ax = plt.subplots(figsize=(7, 5))
ns = np.array(list(NS), float)
ax.plot(ns, 6 * ns + 1, "--", color="tab:blue", alpha=.55,
        label=r"deep expected:  $6n+1$")
ax.plot(list(SHALLOW), [v[1] for v in SHALLOW.values()], "o-", color="tab:red",
        ms=8, label="shallow (measured)")
ax.plot(list(DEEP), [v[1] for v in DEEP.values()], "s-", color="tab:blue",
        ms=7, label="deep, $w=2$ (measured)")
ax.plot([n for n in NS if n not in SHALLOW], [385] * 5, "x", color="tab:red",
        ms=11, mew=2.5)
ax.plot([n for n in NS if n not in DEEP], [37, 43, 49], "x", color="tab:blue",
        ms=11, mew=2.5)
ax.annotate("shallow: $w>128$ (no fit)", (4, 250), textcoords="offset points",
            xytext=(-10, 14), color="tab:red", fontsize=9)
ax.annotate("deep: lr too large", (6, 37), textcoords="offset points",
            xytext=(6, -16), color="tab:blue", fontsize=9)
ax.set_yscale("log"); ax.set_xlabel(r"map degree $n$  (target $f^{(n)}$, $2^n$ teeth)")
ax.set_ylabel("parameters at minimum sufficient width")
ax.set_title("code1a: cost of representing the $n$-fold tent map")
ax.grid(alpha=.3, which="both"); ax.legend(loc="center left", fontsize=9)
fig.tight_layout(); fig.savefig("code1a_fig1_params_vs_n.png", dpi=150)

# ---------------- Figure 2: accuracy vs depth ----------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(1 / 12, ls=":", color="k", alpha=.6)
ax.annotate("constant-predictor baseline  Var$(y)=1/12$", (2.05, 1 / 12),
            textcoords="offset points", xytext=(0, 6), fontsize=9)
ax.plot(list(SHAL_MSE), list(SHAL_MSE.values()), "o-", color="tab:red",
        ms=8, label="shallow (best width $\\leq 128$)")
ax.plot(list(DEEP_MSE), list(DEEP_MSE.values()), "s-", color="tab:blue",
        ms=7, label="deep, $w=2$ (13-49 params)")
ax.axhline(1e-3, ls="--", color="grey", alpha=.7)
ax.annotate("tol $=10^{-3}$", (7.4, 1e-3), textcoords="offset points",
            xytext=(0, 6), fontsize=9, color="grey")
ax.set_yscale("log"); ax.set_xlabel(r"map degree $n$")
ax.set_ylabel("dense-grid MSE (best of 5 seeds)")
ax.set_title("code1a: accuracy vs depth of the map")
ax.grid(alpha=.3, which="both"); ax.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig("code1a_fig2_mse_vs_n.png", dpi=150)

# ---------------- Figure 3: what the two actually learn ----------------
xg = torch.linspace(0, 1, 2001).unsqueeze(1)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
for col, n in enumerate((3, 4, 6)):
    yt = code1a.tent_map_n(xg.squeeze(1).numpy(), n)
    for row, (kind, w) in enumerate((("shallow", 128), ("deep", 2))):
        r = code1a.train_best(kind, w=w, n=n, epochs=800)
        with torch.no_grad():
            yp = r["model"](xg).squeeze(1).numpy()
        a = axes[row, col]
        a.plot(xg.squeeze(1), yt, color="k", lw=1.2, alpha=.65, label="target $f^{(n)}$")
        a.plot(xg.squeeze(1), yp, color="tab:red" if row == 0 else "tab:blue",
               lw=1.1, label=f"{kind} $w$={w}")
        a.set_title(f"{kind}, $n$={n}, {r['n_params']} params\n"
                    f"MSE {r['grid_mse']:.1e}, {r['pieces']} pieces (target {2**n})",
                    fontsize=9)
        a.grid(alpha=.25)
        if col == 0: a.set_ylabel("$f(x)$")
        if row == 1: a.set_xlabel("$x$")
        a.legend(fontsize=7, loc="upper right")
fig.suptitle("code1a: shallow (top) vs deep $w=2$ (bottom) -- what is actually learned",
             fontsize=12)
fig.tight_layout(); fig.savefig("code1a_fig3_functions.png", dpi=150)
print("wrote code1a_fig1_params_vs_n.png, code1a_fig2_mse_vs_n.png, code1a_fig3_functions.png")
