"""Helpers for the mini course "Machine Learning for Mathematics".

Everything here is support code: building the dataset, encoding numbers,
plotting, and the probes used to look inside a trained network. The training
loop is written out in notebooks/01_train.ipynb, and the same loop is repeated
here as train() so the explore notebook and tools/ can reuse it.
"""
import base64
import copy
import io
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- style

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#9ec5f4", "#2a78d6", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_orange", ["#eb6834", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})

# ---------------------------------------------------------------- data


def digits_of(numbers, n_digits=4):
    """Digits of each number, most significant first. 407 -> [0, 4, 0, 7]."""
    numbers = torch.as_tensor(numbers)
    return torch.stack([(numbers // 10 ** k) % 10 for k in reversed(range(n_digits))], dim=1)


def make_data(modulus=6, n_digits=4, train_frac=0.5, seed=0, holdout=None):
    """All numbers with n_digits digits, labeled by n mod modulus, split at random.

    holdout=(position, digit) removes every number with that digit in that
    position (0 = leftmost) from training and sets those numbers aside in
    data.held, so you can test the network on a digit it has never seen there.
    """
    numbers = torch.arange(10 ** n_digits)
    order = torch.randperm(len(numbers), generator=torch.Generator().manual_seed(seed))
    n_train = int(train_frac * len(numbers))
    train, test = order[:n_train], order[n_train:]
    held = torch.empty(0, dtype=torch.long)
    if holdout is not None:
        position, digit = holdout
        has_digit = digits_of(numbers, n_digits)[:, position] == digit
        held, train, test = test[has_digit[test]], train[~has_digit[train]], test[~has_digit[test]]
    return SimpleNamespace(numbers=numbers, labels=numbers % modulus, train=train, test=test,
                           held=held, modulus=modulus, n_digits=n_digits)


def encode_raw(numbers, n_digits=4):
    """One input: the number itself, scaled to lie between 0 and 1."""
    return (torch.as_tensor(numbers).float() / 10 ** n_digits).unsqueeze(1)


def encode_digits(numbers, n_digits=4):
    """One input per digit: the digit's value, scaled to lie between 0 and 1."""
    return digits_of(numbers, n_digits).float() / 9


def encode_onehot(numbers, n_digits=4):
    """Ten inputs per digit: a 1 in the slot for that digit and 0 elsewhere."""
    onehot = nn.functional.one_hot(digits_of(numbers, n_digits), num_classes=10)
    return onehot.float().reshape(len(onehot), -1)


# ---------------------------------------------------------------- model and training


def make_model(n_inputs, width, n_classes):
    """One hidden layer: inputs -> width hidden units (ReLU) -> one score per class."""
    return nn.Sequential(nn.Linear(n_inputs, width), nn.ReLU(), nn.Linear(width, n_classes))


def divisors(modulus):
    """Proper divisors bigger than 1. For 6 these are 2 and 3."""
    return [d for d in range(2, modulus) if modulus % d == 0]


@torch.no_grad()
def evaluate(model, X, y, data, epoch=0):
    """Loss and accuracy on the training and test numbers, as one row of history.

    Besides exact accuracy, it records for each divisor d of the modulus how
    often the prediction is right mod d. For n mod 6 that is parity (d = 2)
    and the residue mod 3.
    """
    loss_fn = nn.CrossEntropyLoss()
    out_train, out_test = model(X[data.train]), model(X[data.test])
    pred, truth = out_test.argmax(1), y[data.test]
    row = {
        "epoch": epoch,
        "train_loss": loss_fn(out_train, y[data.train]).item(),
        "test_loss": loss_fn(out_test, truth).item(),
        "train_acc": (out_train.argmax(1) == y[data.train]).float().mean().item(),
        "test_acc": (pred == truth).float().mean().item(),
    }
    for d in divisors(data.modulus):
        row[f"mod{d}_acc"] = (pred % d == truth % d).float().mean().item()
    return row


def train(encode, data, width=512, optimizer="sgd", lr=0.1, epochs=400, batch_size=64,
          seed=0, weight_decay=0.0, save_at=(), live=True, title=None):
    """Train a one-hidden-layer network. Returns (model, history, saved).

    history is a list with one row of measurements per epoch. saved maps each
    epoch in save_at to a frozen copy of the model at that moment.
    This is the loop from notebooks/01_train.ipynb with a few extra options.
    Pressing stop ends training early and keeps the network as it is.
    """
    torch.manual_seed(seed)
    X, y = encode(data.numbers, data.n_digits), data.labels
    model = make_model(X.shape[1], width, data.modulus)
    model.encode = encode
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history, saved = [], {}
    try:
        for epoch in range(epochs + 1):
            history.append(evaluate(model, X, y, data, epoch))
            if epoch in save_at:
                saved[epoch] = copy.deepcopy(model)
            if live:
                live_plot(history, data.modulus, title, done=(epoch == epochs))
            if epoch == epochs:
                break
            order = data.train[torch.randperm(len(data.train))]
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                loss = loss_fn(model(X[batch]), y[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()
    except KeyboardInterrupt:  # the stop button: keep what we have so far
        if live:
            live_plot(history, data.modulus, title, done=True)
        print(f"Stopped early at epoch {history[-1]['epoch']}.")
    return model, history, saved


def summarize(history):
    """One line: where the run ended."""
    last = history[-1]
    parts = [f"epoch {last['epoch']}", f"train accuracy {last['train_acc']:.1%}",
             f"test accuracy {last['test_acc']:.1%}"]
    parts += [f"right mod {k[3:-4]}: {v:.1%}" for k, v in last.items() if k.startswith("mod")]
    print(" | ".join(parts))

# ---------------------------------------------------------------- saving and loading


def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path, encode=encode_onehot):
    """Rebuild a saved network. Its shape is read off the saved weights."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch without the weights_only option
        state = torch.load(path, map_location="cpu")
    width, n_inputs = state["0.weight"].shape
    model = make_model(n_inputs, width, state["2.weight"].shape[0])
    model.load_state_dict(state)
    model.encode = encode
    return model.eval()


def save_history(history, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(history[0])
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            f.write(",".join(f"{row[k]:.6g}" for k in keys) + "\n")


def load_history(path):
    with open(path) as f:
        keys = f.readline().strip().split(",")
        rows = [dict(zip(keys, map(float, line.strip().split(",")))) for line in f if line.strip()]
    for row in rows:
        row["epoch"] = int(row["epoch"])
    return rows

# ---------------------------------------------------------------- plots of training


def _column(history, key):
    return [row[key] for row in history]


def plot_history(history, modulus=6, title=None, marks=()):
    """Loss (top) and accuracy (bottom) against epochs, on a stretched epoch axis.

    The dotted lines in the loss panel sit at ln(k). A network that has narrowed
    every number down to k equally likely answers has loss exactly ln(k).
    marks is a list of epochs to flag with a vertical line.
    """
    epochs = _column(history, "epoch")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    top.plot(epochs, _column(history, "test_loss"), color=BLUE, label="loss on test numbers")
    top.plot(epochs, _column(history, "train_loss"), color=BLUE, alpha=0.45, linestyle="--",
             label="loss on training numbers")
    right = max(epochs[-1], 1)
    levels = sorted({modulus} | {modulus // d for d in divisors(modulus)})
    for k in levels:
        top.axhline(math.log(k), color=MUTED, linewidth=1, linestyle=":")
    top.set_yticks([0] + [math.log(k) for k in levels])
    top.set_yticklabels(["0"] + [f"ln {k} = {math.log(k):.2f}" for k in levels])
    top.grid(axis="y", visible=False)
    top.set_ylabel("loss")
    highest = max(max(row["test_loss"], row["train_loss"]) for row in history)
    top.set_ylim(0, max(math.log(modulus) * 1.12, highest * 1.05))
    top.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    top.set_title(title or f"Learning n mod {modulus}", loc="left")

    bottom.plot(epochs, _column(history, "test_acc"), color=BLUE, label=f"exactly right (mod {modulus})")
    for color, d in zip(SERIES[1:], divisors(modulus)):
        name = "parity right (mod 2)" if d == 2 else f"right mod {d}"
        bottom.plot(epochs, _column(history, f"mod{d}_acc"), color=color, label=name)
    bottom.plot(epochs, _column(history, "train_acc"), color=BLUE, alpha=0.45, linestyle="--",
                label="exactly right, training numbers")
    bottom.set_ylabel("accuracy on test numbers")
    bottom.set_ylim(-0.03, 1.03)
    bottom.set_yticks([0, 0.25, 0.5, 0.75, 1])
    bottom.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    bottom.set_xlabel("epochs of training (stretched scale)")
    bottom.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    for ax in (top, bottom):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, right)
        for m in marks:
            ax.axvline(m, color=INK, linewidth=1, alpha=0.5)
    ticks = [t for t in (0, 1, 3, 10, 30, 100, 300, 1000) if t <= right]
    bottom.set_xticks(ticks)
    bottom.set_xticklabels([str(t) for t in ticks])
    return fig


LIVE_SECONDS = 1.5  # at most one redraw this often, so the picture never flashes
LIVE_SIZE = (9.5, 6.4)  # inches; every frame has exactly this size, so nothing on the page jumps
LIVE_DPI = 90
_live = {"handle": None, "last": 0.0, "image": None}


def _frame(history, modulus, title):
    """The training plot as PNG bytes, always the same pixel size."""
    fig = plot_history(history, modulus, title)
    fig.set_size_inches(*LIVE_SIZE)
    fig.subplots_adjust(left=0.15, right=0.71, top=0.94, bottom=0.09)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=LIVE_DPI)
    plt.close(fig)
    return buffer.getvalue()


def live_plot(history, modulus=6, title=None, done=False):
    """Show the training plot and update it in place as training goes on.

    Call it every epoch. It redraws at most once every LIVE_SECONDS seconds,
    always draws the first and the last frame, and never clears the output,
    so the picture changes smoothly instead of flashing.
    """
    first = len(history) == 1
    now = time.time()
    if not (first or done or now - _live["last"] >= LIVE_SECONDS):
        return
    _live["last"] = now
    png = _frame(history, modulus, title)
    if _live["image"] is not None:  # the explore panel supplies its own picture widget
        _live["image"].value = png
        return
    from IPython.display import HTML, display
    width, height = int(LIVE_SIZE[0] * LIVE_DPI), int(LIVE_SIZE[1] * LIVE_DPI)
    picture = HTML(f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}" '
                   f'width="{width}" height="{height}" style="max-width:100%;height:auto">')
    if first or _live["handle"] is None:
        _live["handle"] = display(picture, display_id=True)
    else:
        _live["handle"].update(picture)

# ---------------------------------------------------------------- probes


@torch.no_grad()
def predict_all(model, numbers, n_digits=4):
    """The network's answer for each number."""
    return model(model.encode(torch.as_tensor(numbers), n_digits)).argmax(1)


@torch.no_grad()
def predict(model, n, modulus=None, n_digits=4, ax=None, label=None):
    """Ask the network about one number and draw how sure it is of each answer."""
    probs = torch.softmax(model(model.encode(torch.tensor([n]), n_digits)), dim=1)[0]
    modulus = modulus or len(probs)
    guess, truth = int(probs.argmax()), n % modulus
    own_axis = ax is None
    if own_axis:
        _, ax = plt.subplots(figsize=(5, 2.4))
    colors = [AQUA if r == truth else MUTED for r in range(modulus)]
    ax.bar(range(modulus), probs.numpy(), color=colors, width=0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(modulus))
    ax.set_xlabel("answer (green = correct)")
    ax.set_ylabel("confidence")
    verdict = "correct" if guess == truth else f"wrong, truth is {truth}"
    ax.set_title(f"{label + ': ' if label else ''}{n:0{n_digits}d} -> says {guess} ({verdict})",
                 loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    if own_axis:
        plt.show()
    return guess


def ask(models, n, n_digits=4):
    """Ask several networks about the same number. models is a dict {name: model}."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 2.7), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        predict(model, n, n_digits=n_digits, ax=ax, label=name)
    fig.tight_layout()
    plt.show()


def wrong_numbers(model, data):
    """The test numbers this network gets wrong, in increasing order."""
    numbers = data.numbers[data.test]
    wrong = predict_all(model, numbers, data.n_digits) != data.labels[data.test]
    return sorted(numbers[wrong].tolist())


def confusion(model, data):
    """table[t, p] = how many test numbers with true answer t got prediction p."""
    pred, truth = predict_all(model, data.numbers[data.test], data.n_digits), data.labels[data.test]
    table = torch.zeros(data.modulus, data.modulus, dtype=torch.long)
    for t, p in zip(truth, pred):
        table[t, p] += 1
    return table


def _heatmap(ax, table, cmap, vmin, vmax, fmt, labels):
    table = np.asarray(table, dtype=float)
    ax.imshow(table, cmap=cmap, vmin=vmin, vmax=vmax)
    middle = (vmin + vmax) / 2
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            far = abs(table[i, j] - middle) > 0.33 * (vmax - vmin)
            dark = far if cmap is DIVERGING else table[i, j] > middle
            ax.text(j, i, fmt(table[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.grid(False)
    ax.tick_params(length=0)


def plot_confusions(models, data):
    """One confusion table per model. models is a dict {name: model}."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.4), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        table = confusion(model, data)
        accuracy = table.diag().sum().item() / table.sum().item()
        _heatmap(ax, table, SEQUENTIAL, 0, table.sum(1).max().item(), lambda v: f"{int(v)}" if v else "",
                 range(data.modulus))
        ax.set_title(f"{name}: {accuracy:.0%} right", loc="left")
        ax.set_xlabel("network's answer")
        ax.set_ylabel(f"true n mod {data.modulus}")
    fig.tight_layout()
    plt.show()


POSITION_NAMES = {4: ["thousands", "hundreds", "tens", "ones"]}


def position_name(position, n_digits=4):
    return POSITION_NAMES.get(n_digits, [f"position {i}" for i in range(n_digits)])[position]


@torch.no_grad()
def swap_test(model, data, position, shift):
    """Add shift to one digit and see whether the network changes its answer.

    Uses every number whose digit in that position can take the shift
    without leaving 0..9. Returns the fraction of answers left unchanged.
    """
    numbers = data.numbers
    digit = digits_of(numbers, data.n_digits)[:, position]
    ok = (digit + shift >= 0) & (digit + shift <= 9)
    numbers = numbers[ok]
    swapped = numbers + shift * 10 ** (data.n_digits - 1 - position)
    same = predict_all(model, numbers, data.n_digits) == predict_all(model, swapped, data.n_digits)
    return same.float().mean().item()


@torch.no_grad()
def swap_table(model, data, position):
    """table[a, b] = how often the answer survives replacing digit a by digit b."""
    numbers = data.numbers
    digit = digits_of(numbers, data.n_digits)[:, position]
    place = 10 ** (data.n_digits - 1 - position)
    before = predict_all(model, numbers, data.n_digits)
    table = torch.ones(10, 10)
    for a in range(10):
        chosen = numbers[digit == a]
        for b in range(10):
            after = predict_all(model, chosen + (b - a) * place, data.n_digits)
            table[a, b] = (after == before[digit == a]).float().mean()
    return table


def plot_swap_tables(models, data, position):
    """Swap tables for several models, side by side. models is {name: model}."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.7), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        _heatmap(ax, swap_table(model, data, position), SEQUENTIAL, 0, 1,
                 lambda v: f"{round(100 * v)}", range(10))
        ax.set_title(f"{name}", loc="left")
        ax.set_xlabel("new digit")
        ax.set_ylabel(f"original {position_name(position, data.n_digits)} digit")
    fig.suptitle(f"How often (%) the answer survives changing the {position_name(position, data.n_digits)} digit",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def digit_similarity(model, position, n_digits=4):
    """How alike the network's first layer treats each pair of digits in one position.

    Each digit in each position has its own column of first-layer weights, one
    weight per hidden unit. This returns the cosine similarity between those
    columns: +1 means two digits push the hidden units the same way, -1 means
    opposite ways. Only meaningful for the one-hot encoding.
    """
    if not isinstance(position, int):  # several positions: average their tables
        return sum(digit_similarity(model, p, n_digits) for p in position) / len(position)
    weights = model[0].weight.reshape(-1, n_digits, 10)[:, position, :]
    weights = weights - weights.mean(dim=1, keepdim=True)
    weights = weights / weights.norm(dim=0, keepdim=True)
    return weights.T @ weights


def plot_similarities(models, positions=(0, 3), n_digits=4):
    """Digit similarity for each model (rows) and digit position (columns)."""
    fig, axes = plt.subplots(len(models), len(positions),
                             figsize=(4.6 * len(positions), 4.5 * len(models)), squeeze=False)
    for row, (name, model) in zip(axes, models.items()):
        for ax, position in zip(row, positions):
            _heatmap(ax, digit_similarity(model, position, n_digits), DIVERGING, -0.5, 0.5,
                     lambda v: f"{v:+.1f}" if abs(v) < 0.995 else "",
                     range(10))
            where = (f"{position_name(position, n_digits)} digit" if isinstance(position, int)
                     else "other digits (averaged)")
            ax.set_title(f"{name}: {where}", loc="left")
    fig.tight_layout()
    plt.show()
