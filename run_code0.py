"""Benchmark the ORIGINAL code (code0.py) on the same footing as code1/code1a.

The MEASUREMENT is upgraded to match code1*.py: instead of the original random test split (which is statistically identical to its training set), every model is scored on the dense grid linspace(0,1,20001), with sup-norm and a piece count.

NOTE: code0 feeds the network 2x-1, so the grid must be rescaled the same way.
"""
import numpy as np, torch, code0

GRID = torch.linspace(0.0, 1.0, 20001).unsqueeze(1)
WIDTHS = (2, 4, 8, 16, 32, 64, 128)
EPOCHS, TOL = 20_000, 1e-3


def grid_scores(model, n):
    y = torch.tensor(code0.iterative_map(code0.tent_map, GRID.squeeze(1).numpy(), n),
                     dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        p = model(2.0 * GRID - 1.0)              # code0's input convention
    mse = ((p - y) ** 2).mean().item()
    sup = (p - y).abs().max().item()
    d2 = (p.squeeze(1)[2:] - 2 * p.squeeze(1)[1:-1] + p.squeeze(1)[:-2]).abs()
    tol = max(0.25 * d2.max().item(), 1e-9)
    hit = d2 > tol
    st = hit.clone(); st[1:] &= ~hit[:-1]
    return mse, sup, int(st.sum()) + 1


print(f"=== code0 (original upload) | lr=1e-3 full batch, epochs={EPOCHS}, "
      f"1 seed, deep w=2 ===")
print(f"    scored on the dense grid; tol = {TOL:.0e}\n")
print(f"{'n':>2} {'kind':>8} {'min w':>7} {'params':>7} {'gridMSE':>10} "
      f"{'sup':>10} {'pieces':>7} {'2^n':>5}")

for n in range(2, 9):
    for kind in ("shallow", "deep"):
        search = (2,) if kind == "deep" else WIDTHS
        hit = None
        for w in search:
            r = code0.train(kind=kind, w=w, n=n, epochs=EPOCHS, log_every=EPOCHS + 1)
            mse, sup, pieces = grid_scores(r["model"], n)
            last = (w, r, mse, sup, pieces)
            if mse < TOL:
                hit = last
                break
        w, r, mse, sup, pieces = hit if hit else last
        tag = f"{w}" if hit else ("w > 128" if kind == "shallow" else "fail")
        print(f"{n:>2} {kind:>8} {tag:>7} {r['n_params'] if hit else '':>7} "
              f"{mse:>10.3e} {sup:>10.3e} {pieces:>7} {2**n:>5}")
    print()
