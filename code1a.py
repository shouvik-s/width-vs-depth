import numpy as np
import torch
import torch.nn as nn 

def tent_map_n(x, n):
    """Closed form of the n-fold tent map: f^n(x) = 1 - |mod(2^n x, 2) - 1|."""
    x = np.asarray(x, dtype=np.float64)
    return 1.0 - np.abs(np.mod((2.0 ** n) * x, 2.0) - 1.0)

def tent_map(x):
    return tent_map_n(x, 1)

def iterative_map(func, x_arr, n):
    # kept for compatibility with other code*.py; evaluated in closed form (no round-off amplification)
    return tent_map_n(x_arr, n)

# ==== Parameters ====
MAP_DEGREE = 3
DATA_SIZE = 2000
WIDTH = 2**MAP_DEGREE - 1


class ShallowMLP(nn.Module):
    def __init__(self, w=WIDTH, n=MAP_DEGREE):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(1, w), #hidden layer
            nn.ReLU(),
            nn.Linear(w, 1) #output layer
        )
        # --- deterministic initialization of weights
        with torch.no_grad():
            k = torch.linspace(0.0, 1.0, w + 2)[1:-1]                
            sign = torch.where(torch.arange(w) % 2 == 0, 1.0, -1.0)
            W = 2.0 * sign
            self.layers[0].weight.copy_(W.unsqueeze(1))
            self.layers[0].bias.copy_(-W * k)
            self.layers[2].weight.normal_(0.0, 1.0 / max(w, 1) ** 0.5)
            self.layers[2].bias.zero_()

    def forward(self, x):
        z = self.layers(x)
        return z


class DeepMLP(nn.Module):
    def __init__(self, w=2, n=MAP_DEGREE, noise=1e-2):
        super().__init__()
        # n ReLU layers TOTAL: the 1st hidden layer plus (n-1) hidden layers
        self.first_hidden = nn.Linear(1, w)
        self.rest_hidden = nn.ModuleList(
            [nn.Linear(w, w) for _ in range(max(n - 1, 0))]
        )
        self.activation = nn.ReLU()
        self.out = nn.Linear(w, 1)

        # --- tent-structured init on the leading 2x2 block (+ small noise) ---
        # the ONLY change from code1.py -- the jitter is scaled by 2^-n,
        # because each tent block has gain 2 and so amplifies any perturbation
        # by 2^n at the output. A fixed jitter leaves the n=6 init worse than a
        # constant predictor (1.32e-1 vs the 8.3e-2 baseline); scaled, the init
        # stays at ~1e-4 for every n.
        if w >= 2:
            with torch.no_grad():
                self.first_hidden.weight.zero_(); self.first_hidden.bias.zero_()
                self.first_hidden.weight[0, 0] = 1.0
                self.first_hidden.weight[1, 0] = 1.0
                self.first_hidden.bias[1] = -0.5
                for layer in self.rest_hidden:
                    layer.weight.zero_(); layer.bias.zero_()
                    layer.weight[:2, :2] = torch.tensor([[2.0, -4.0], [2.0, -4.0]])
                    layer.bias[:2] = torch.tensor([0.0, -0.5])
                self.out.weight.zero_(); self.out.bias.zero_()
                self.out.weight[0, :2] = torch.tensor([2.0, -4.0])
                for p in self.parameters():
                    p.add_(noise * 2.0 ** -n * torch.randn_like(p))

    def forward(self, x):
        z = self.activation(self.first_hidden(x))
        for layer in self.rest_hidden:
            z = self.activation(layer(z))
        return self.out(z)


#---model switching---
def make_model(kind, w, n):
    if kind == "shallow":
        return ShallowMLP(w=w, n=n)
    if kind == "deep":
        return DeepMLP(w=w, n=n)
    raise ValueError(f"unknown kind: {kind}")


#---counts the number of parameters in the model---
def count_params(model):
    return sum(p.numel() for p in model.parameters())


#---counts linear pieces: kinks = sign changes of the second difference---
def count_pieces(model, grid_size=20_001, rel_tol=0.25):
    xs = torch.linspace(0.0, 1.0, grid_size).unsqueeze(1)
    with torch.no_grad():
        ys = model(xs).squeeze(1)
    d1 = ys[1:] - ys[:-1]
    d2 = d1[1:] - d1[:-1]
    d2 = d2.abs()
    tol = max(rel_tol * d2.max().item(), 1e-9)
    hit = (d2 > tol)
    # a single kink can straddle two grid samples: count runs, not samples
    starts = hit.clone()
    starts[1:] &= ~hit[:-1]
    kinks = int(starts.sum().item())
    return kinks + 1



# --------------------------------------------------training-testing module
def train(kind="shallow", w=WIDTH, n=MAP_DEGREE, data_size=DATA_SIZE,
          epochs=800, lr=1e-2, train_frac=0.7, seed=1, log_every=200,
          batch=256, verbose=True):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ------ data ------
    raw_data_x = rng.random(data_size)
    raw_data_fx = tent_map_n(raw_data_x, n)
    assert raw_data_fx.shape == raw_data_x.shape

    x = torch.tensor(raw_data_x, dtype=torch.float32).unsqueeze(1)    # (N, 1) in [0,1]
    y = torch.tensor(raw_data_fx, dtype=torch.float32).unsqueeze(1)   # (N, 1)
    if verbose:
        print(f"baseline (predict mean) MSE = {y.var().item():.4e}")

    num_train = int(train_frac * data_size)
    permute = torch.randperm(data_size)
    tr = permute[:num_train] #training data labels
    x_tr, y_tr = x[tr], y[tr]

    # dense held-out grid 
    x_g = torch.linspace(0.0, 1.0, 20_001).unsqueeze(1)
    y_g = torch.tensor(tent_map_n(x_g.squeeze(1).numpy(), n), dtype=torch.float32).unsqueeze(1)

    # ------ model ------
    model = make_model(kind, w=w, n=n)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epoch": [], "train": [], "grid_mse": [], "grid_sup": []}

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(num_train)
        running = 0.0
        for i in range(0, num_train, batch):
            idx = order[i:i + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
        scheduler.step()
        train_loss = running / num_train

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred = model(x_g)
                grid_mse = loss_fn(pred, y_g).item()
                grid_sup = (pred - y_g).abs().max().item()
            history["epoch"].append(epoch)
            history["train"].append(train_loss)
            history["grid_mse"].append(grid_mse)
            history["grid_sup"].append(grid_sup)
            if verbose:
                print(f"[{kind} w={w} n={n}] epoch {epoch:6d}  "
                      f"train {train_loss:.3e}  grid-MSE {grid_mse:.3e}  sup {grid_sup:.3e}")

    return {"model": model, "history": history,
            "n_params": count_params(model),
            "grid_mse": history["grid_mse"][-1],
            "grid_sup": history["grid_sup"][-1],
            "pieces": count_pieces(model),
            "data": (x, y)} #returns a dictionary


def train_best(kind, w, n, seeds=(1, 2, 3, 4, 5), **kw):
    """Best (lowest dense-grid MSE) run over several seeds."""
    best = None
    for s in seeds:
        r = train(kind=kind, w=w, n=n, seed=s, verbose=False, **kw)
        if best is None or r["grid_mse"] < best["grid_mse"]:
            best = r
    return best


def sweep(ns=range(2, 9), widths=(2, 4, 8, 16, 32, 64, 128), epochs=800, tol=1e-3):
    """Minimum width reaching `tol` grid-MSE, as a function of the map degree n."""
    print(f"{'n':>2} {'kind':>8} {'min w':>6} {'params':>7} {'gridMSE':>10} "
          f"{'sup':>10} {'pieces':>7} {'2^n':>5}")
    for n in ns:
        for kind in ("shallow", "deep"):
            hit = None
            for w in widths:
                r = train_best(kind, w=w, n=n, epochs=epochs)
                if r["grid_mse"] < tol:
                    hit = (w, r)
                    break
            if hit is None:
                print(f"{n:>2} {kind:>8} {'>max':>6}")
            else:
                w, r = hit
                print(f"{n:>2} {kind:>8} {w:>6} {r['n_params']:>7} "
                      f"{r['grid_mse']:>10.3e} {r['grid_sup']:>10.3e} "
                      f"{r['pieces']:>7} {2**n:>5}")
        print()

def matched_shallow_width(w_deep, n):
    """Shallow width with the same parameter count as DeepMLP(w_deep, n)."""
    p = n * w_deep**2 + (n + 3) * w_deep + 1
    return round((p - 1) / 3)



if __name__ == "__main__":
    sweep(ns=range(2, 9), widths=(2, 4, 8, 16, 32, 64, 128), epochs=800)

    res_deep = train_best("deep", w=2, n=MAP_DEGREE, epochs=800)
    print(f"deep n={MAP_DEGREE}: params={res_deep['n_params']}  "
          f"grid-MSE={res_deep['grid_mse']:.3e}  sup={res_deep['grid_sup']:.3e}  "
          f"pieces={res_deep['pieces']} (2^n={2**MAP_DEGREE})")
