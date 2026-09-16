import numpy as np
import torch
import torch.nn as nn 

def tent_map(x):
    assert np.all((x >= 0) & (x <= 1)), ("tent-map only applies in the range [0,1]")
    return 1.0 - np.abs(1-2*x)


def iterative_map(func, x_arr, n):
    x = np.copy(x_arr)
    for _ in range(n):
        x = func(x)
    return x

# ==== Parameters ====
MAP_DEGREE = 3
DATA_SIZE = 2000
WIDTH = 2**MAP_DEGREE - 1


class ShallowMLP(nn.Module):
    def __init__(self, w=WIDTH, n=MAP_DEGREE):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(1, w),
            nn.ReLU(),
            nn.Linear(w, 1)
        )

    def forward(self, x):
        z = self.layers(x)
        return z


class DeepMLP(nn.Module):
    def __init__(self, w=2, n=MAP_DEGREE):
        super().__init__()
        self.first_hidden = nn.Linear(1, w)
        self.rest_hidden = nn.ModuleList(
            [nn.Linear(w, w) for _ in range(n-1)]
        )
        self.activation = nn.ReLU()
        self.out = nn.Linear(w, 1)

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



# --------------------------------------------------training-testing module
def train(kind="shallow", w=WIDTH, n=MAP_DEGREE, data_size=DATA_SIZE,
          epochs=10_000, lr=1e-3, train_frac=0.7, seed=1, log_every=1_000):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ------ data ------
    raw_data_x = rng.random(data_size)
    raw_data_fx = iterative_map(tent_map, raw_data_x, n)
    assert raw_data_fx.shape == raw_data_x.shape

    x = torch.tensor(2.0*raw_data_x - 1.0, dtype=torch.float32).unsqueeze(1)   # (N, 1)
    y = torch.tensor(raw_data_fx, dtype=torch.float32).unsqueeze(1)   # (N, 1)

    num_train = int(train_frac * data_size)
    permute = torch.randperm(data_size)
    tr, te = permute[:num_train], permute[num_train:] # train, test
    x_tr, y_tr, x_te, y_te = x[tr], y[tr], x[te], y[te]
    
    # ------ model ------
    model = make_model(kind, w=w, n=n)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create dictionary to store progress during epochal evolution of MLP
    history = {"epoch": [], "train": [], "test": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss=loss_fn(model(x_tr), y_tr) #training loss
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(x_te), y_te).item()
            history["epoch"].append(epoch)
            history["train"].append(loss.item())
            history["test"].append(test_loss)
            print(f"[{kind} w={w}] epoch {epoch:6d}  "
                  f"train {loss.item():.3e}  test {test_loss:.3e}")
    
    return {"model": model, "history": history,
            "n_params": count_params(model),
            "data": (x, y)} #returns a dictionary



def sweep(k=MAP_DEGREE, widths=(2, 4, 8, 16, 32, 64, 128), epochs=20_000):
    for w in widths:
        r = train(kind="shallow", w=w, n=k, epochs=epochs, log_every=epochs)
        print(f"k={k}  w={w:4d}  params={r['n_params']:6d}  "
              f"test={r['history']['test'][-1]:.3e}")
        print("\n")

def matched_shallow_width(w_deep, n):
    """Shallow width with the same parameter count as DeepMLP(w_deep, n)."""
    p = n * w_deep**2 + (n + 3) * w_deep + 1
    return round((p - 1) / 3)
