"""Comparison table for code1a.py (= code1.py with the 2^-n noise scaling).

    lr      = 1e-2   (code1.py's default, unchanged, for BOTH architectures)
    epochs  = 800    (code1.py's default)
    deep    : width fixed at 2
    shallow : width swept 2..128
    tol     = 1e-3   (not specified in the request; stated in the header)
    seeds   = 5, best of
"""
import code1a

EPOCHS, TOL = 800, 1e-3
WIDTHS = (2, 4, 8, 16, 32, 64, 128)
SEEDS = (1, 2, 3, 4, 5)

print(f"=== code1a: code1 + 2^-n noise scaling | lr=1e-2 (code1 default), "
      f"epochs={EPOCHS}, deep w=2 fixed ===")
print(f"    tol = {TOL:.0e}, best of {len(SEEDS)} seeds\n")
print(f"{'n':>2} {'kind':>8} {'min w':>6} {'params':>7} {'gridMSE':>10} "
      f"{'sup':>10} {'pieces':>7} {'2^n':>5}")

for n in range(2, 9):
    for kind in ("shallow", "deep"):
        search = (2,) if kind == "deep" else WIDTHS
        hit = None
        for w in search:
            r = code1a.train_best(kind, w=w, n=n, epochs=EPOCHS, seeds=SEEDS)
            if r["grid_mse"] < TOL:
                hit = (w, r)
                break
        if hit is None:
            last = r
            print(f"{n:>2} {kind:>8} {'fail':>6} {'':>7} {last['grid_mse']:>10.3e} "
                  f"{last['grid_sup']:>10.3e} {last['pieces']:>7} {2**n:>5}")
        else:
            w, r = hit
            print(f"{n:>2} {kind:>8} {w:>6} {r['n_params']:>7} "
                  f"{r['grid_mse']:>10.3e} {r['grid_sup']:>10.3e} "
                  f"{r['pieces']:>7} {2**n:>5}")
    print()
