"""Helpers for the mini course "Machine Learning for Mathematics".

Everything here is support code: building the dataset, encoding numbers,
plotting, the checks behind the fill-in cells, and the probes used to look
inside a trained network. The training loop is written out in
notebooks/01_train.ipynb, and the same loop is repeated here as train() so the
lab.py and tools/ can reuse it.

In the notebooks, a ... (three dots) marks a blank for students to fill in.
Every function a student calls reads what they typed loosely (2, 2.0, "2" and a
stray trailing comma all mean 2) and prints a plain message instead of crashing
when it cannot. Importing this module also adds a "What this probably means"
hint under every error in the notebook, syntax errors included.
"""
import base64
import builtins
import copy
import difflib
import functools
import html
import io
import math
import os
import re
import sys
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
EDGE, OFF, LIGHT_BLUE = "#5f5e58", "#f1f0eb", "#9ec5f4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#9ec5f4", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


# ---------------------------------------------------------------- reading what students type


class Unreadable(Exception):
    """Something a student typed that a helper cannot use. The message is meant for the student."""


def _blank(value):
    """True if a student left a ... blank unfilled."""
    return value is Ellipsis


def _forgiving(function):
    """Print an Unreadable message instead of raising it, and return None."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Unreadable as problem:
            print(problem)
            return None
    return wrapper


def _unwrap(value):
    """A stray comma turns 2 into (2,). Undo that."""
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    return value


def _as_number(value, what="a number"):
    """Read what a student typed as a number: 50, 50.0, "50", "50%", (50,) all work."""
    value = _unwrap(value)
    if _blank(value):
        raise Unreadable(f"Replace the ... with {what} and run the cell again.")
    if isinstance(value, bool) or value is None:
        raise Unreadable(f"Type {what} here, like 3, not {value!r}.")
    if isinstance(value, str):
        text = value.replace(" ", "").replace("_", "").strip(",;").rstrip("%")
        if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+", text):  # 1,999 as one number
            text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            raise Unreadable(f"Could not read {value!r} as {what}. Type digits only, like 3.") from None
    try:
        number = float(value)  # ints, floats, numpy and torch numbers
    except (TypeError, ValueError):
        raise Unreadable(f"Could not read {value!r} as {what}. Type digits only, like 3.") from None
    if not math.isfinite(number):
        raise Unreadable(f"{value!r} is not {what}. Type digits only, like 3.")
    return number


def _as_int(value, what="a whole number"):
    """Read what a student typed as a whole number. 2, 2.0, "2" and (2,) all give 2."""
    number = _as_number(value, what)
    if number != int(number):
        raise Unreadable(f"{_unwrap(value)!r} is not a whole number. Type it without a decimal point, "
                         "and without commas or periods inside the number.")
    return int(number)


def _flatten(values):
    """Turn whatever a student typed into a flat list: 7, [7, 8], [[7], 8], "7, 8", range(3)..."""
    if isinstance(values, str):
        tokens = []
        for token in re.split(r"[\s;]+", values.strip()):
            if re.fullmatch(r"\d{1,3}(,\d{3})+", token):
                tokens.append(token.replace(",", ""))  # 1,999 is one number
            else:
                tokens += [t for t in token.split(",") if t]
        return tokens
    if isinstance(values, dict):
        values = list(values.values())
    if isinstance(values, (list, tuple, set, frozenset, range)) or hasattr(values, "tolist"):
        items = values.tolist() if hasattr(values, "tolist") else list(values)
        if not isinstance(items, list):  # a 0-d array or tensor
            return [items]
        flat = []
        for item in (sorted(items) if isinstance(values, (set, frozenset)) else items):
            flat += _flatten(item) if isinstance(item, (list, tuple, set, frozenset, range)) else [item]
        return flat
    return [values]


def _as_models(models):
    """A dict {name: network}, or a single network, which gets the name 'this network'."""
    if models is None:
        raise Unreadable("There's no network here yet: the cell that trains it didn't finish. "
                         "Read the message it printed, fix it, run it again, then run this cell.")
    if isinstance(getattr(models, "model", None), nn.Module):  # the result of lab.run(...)
        models = models.model
    if isinstance(models, nn.Module):
        return {"this network": models}
    if isinstance(models, dict) and models and all(isinstance(m, nn.Module) for m in models.values()):
        return {str(name): model for name, model in models.items()}
    raise Unreadable("The first thing in the brackets should be the networks, like models "
                     "or models[\"epoch 400\"]. Compare with Show answer under the cell. If models was "
                     "changed by accident, run the cell near the top that loads the snapshots again.")


def _as_data(data):
    """The dataset from make_data(), or a clear message if the name data was overwritten."""
    if not all(hasattr(data, key) for key in ("numbers", "labels", "test", "modulus", "n_digits")):
        raise Unreadable("The name data no longer holds the dataset. Run the cell near the top that makes "
                         "it again (it has data = mlmath.make_data(...) in it), then this cell.")
    return data


def _as_history(history):
    """The list of per-epoch rows that train() returns."""
    if history is None:
        raise Unreadable("There's no training run here yet: the line that trains the network didn't finish. "
                         "Read the message it printed, fix it, and run the cell again.")
    if isinstance(getattr(history, "history", None), list):  # the result of lab.run(...)
        history = history.history
    if isinstance(history, tuple) and len(history) == 2 and isinstance(history[1], list):
        history = history[1]  # someone wrote history = train(...) and got (model, history)
    if isinstance(history, nn.Module):
        raise Unreadable("This is a network, not a training history. Check the order of the names: "
                         "model, history = train(...)")
    if not (isinstance(history, list) and history and isinstance(history[0], dict) and "test_acc" in history[0]):
        raise Unreadable("This needs the history from a training run, the second name in "
                         "model, history = train(...). Run the training cell first.")
    return history

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
    """One input per digit: the digit divided by 10, so 4712 becomes 0.4, 0.7, 0.1, 0.2."""
    return digits_of(numbers, n_digits).float() / 10


def encode_onehot(numbers, n_digits=4):
    """Ten inputs per digit: a 1 in the slot for that digit and 0 elsewhere."""
    onehot = nn.functional.one_hot(digits_of(numbers, n_digits), num_classes=10)
    return onehot.float().reshape(len(onehot), -1)


@_forgiving
def show_encodings(*numbers):
    """Print the three ways of showing n to a network."""
    if len(numbers) != 1:
        raise Unreadable("Type one number in the brackets, with no commas inside it: 1999, not 1,999.")
    n = _as_int(numbers[0], "a whole number from 0 to 9999")
    if not 0 <= n <= 9999:
        raise Unreadable(f"{n} is out of range. Pick a whole number from 0 to 9999.")
    x = torch.tensor([n])
    print(f"The number {n:04d}, three ways:\n")
    print("1. Raw value (1 input):      ", [round(v, 4) for v in encode_raw(x)[0].tolist()])
    print("2. Digit values (4 inputs):  ", [round(v, 2) for v in encode_digits(x)[0].tolist()])
    print("3. One-hot digits (40 inputs):")
    for place, row in zip(["thousands", "hundreds", "tens", "ones"], encode_onehot(x)[0].reshape(4, 10)):
        print(f"   {place:9s}", [int(v) for v in row.tolist()])
    _draw_encodings(n)


def _draw_encodings(n):
    """The three encodings of n as input units, shaded by the value each one holds."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = torch.tensor([n])
    names = POSITION_NAMES[4]
    r, row = 0.2, 0.5
    shade = lambda v: SEQUENTIAL(0.05 + 0.75 * v) if v > 0 else "white"
    edge = lambda v: EDGE if v > 0 else MUTED
    heads = [(0.0, "1. Raw value\n1 input"), (2.2, "2. Digit values\n4 inputs"), (7.5, "3. One-hot digits\n40 inputs")]
    for hx, text in heads:
        ax.text(hx - 0.3, 1.25, text, ha="left", va="bottom", fontsize=11, fontweight="bold")
    v = encode_raw(x)[0, 0].item()
    _node(ax, 0.3, 0, r, fill=shade(v), edge=edge(v))
    ax.text(0.3, -0.35, f"{v:g}", ha="center", va="top", fontsize=10)
    for i, v in enumerate(encode_digits(x)[0].tolist()):
        cx = 2.6 + i * 1.1
        ax.text(cx, 0.35, names[i], ha="center", va="bottom", fontsize=8, color=MUTED)
        _node(ax, cx, 0, r, fill=shade(v), edge=edge(v))
        ax.text(cx, -0.35, f"{v:.1f}", ha="center", va="top", fontsize=10)
    for i, column in enumerate(encode_onehot(x)[0].reshape(4, 10).tolist()):
        cx = 8.0 + i * 1.1
        ax.text(cx, 0.35, names[i], ha="center", va="bottom", fontsize=8, color=MUTED)
        for d, v in enumerate(column):
            _node(ax, cx, -d * row, r, fill=shade(v), edge=edge(v))
            ax.text(cx - 0.3, -d * row, str(d), ha="right", va="center", fontsize=7,
                    color=INK if v else MUTED, fontweight="bold" if v else "normal")
    ax.text(-0.3, -4.9, f"What the network is shown for {n:04d}. Darker means a bigger input, and white means 0.",
            ha="left", va="bottom", fontsize=9, color=MUTED)
    ax.set_xlim(-0.4, 11.6)
    ax.set_ylim(-5.0, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    plt.show()


WARM_UP = (1234, 9000, 5555, 2026)


@_forgiving
def check_answers(answers, modulus=6):
    """Compare a student's hand-computed remainders with Python's."""
    if isinstance(answers, (list, tuple)) and len(answers) == len(WARM_UP):
        answers = dict(zip(WARM_UP, answers))
    if not isinstance(answers, dict):
        raise Unreadable("Keep the form {1234: ..., 9000: ..., 5555: ..., 2026: ...} "
                         "and replace only the ... with your answers.")
    for key, guess in answers.items():
        try:
            n = _as_int(key)
        except Unreadable:
            print(f"(skipping {key!r}: it should be one of the four numbers)")
            continue
        truth = n % modulus
        if _blank(guess):
            print(f"{n}: (no answer yet)   Python says {n} % {modulus} = {truth}")
            continue
        try:
            value = _as_int(guess)
        except Unreadable:
            print(f"{n}: could not read your answer {guess!r} as a whole number   "
                  f"Python says {n} % {modulus} = {truth}")
            continue
        mark = "right" if value == truth else "not quite"
        note = "" if 0 <= value < modulus else f" (a remainder mod {modulus} is always 0 to {modulus - 1})"
        print(f"{n}: you said {value}   Python says {n} % {modulus} = {truth}   {mark}{note}")
    print(f"\nIn Python, % means mod: n % {modulus} is the remainder of n after dividing by {modulus}.")

# ---------------------------------------------------------------- model and training


def make_model(n_inputs, width, n_classes, layers=1):
    """inputs -> `layers` hidden layers of `width` units each (ReLU) -> one score per class.

    Session 1's network has one hidden layer.
    """
    parts = [nn.Linear(n_inputs, width), nn.ReLU()]
    for _ in range(layers - 1):
        parts += [nn.Linear(width, width), nn.ReLU()]
    return nn.Sequential(*parts, nn.Linear(width, n_classes))


def divisors(modulus):
    """Proper divisors bigger than 1. For 6 these are 2 and 3."""
    return [d for d in range(2, modulus) if modulus % d == 0]


def _prime_power(d):
    p = next(q for q in range(2, d + 1) if d % q == 0)  # smallest prime factor
    while d % p == 0:
        d //= p
    return d == 1


def pieces(modulus):
    """Divisors of the modulus that are prime powers. For 12 these are 2, 3 and 4.

    The Chinese Remainder Theorem splits n mod m into n mod q for the largest
    prime powers q dividing m; the smaller powers (2 on the way to 4) are steps
    toward those pieces.
    """
    return [d for d in divisors(modulus) if _prime_power(d)]


@torch.no_grad()
def evaluate(model, X, y, data, epoch=0):
    """Loss and accuracy on the training and test numbers, as one row of history.

    Besides exact accuracy, it records for each divisor d of the modulus how
    often the prediction is right mod d. For n mod 6 that is parity (d = 2)
    and the residue mod 3. Session 1 never shows these two columns; session 2
    reveals them at the end.
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
          seed=0, weight_decay=0.0, save_at=(), live="plot", title=None, detail=False, layers=1):
    """Train a one-hidden-layer network. Returns (model, history, saved).

    history is a list with one row of measurements per epoch. saved maps each
    epoch in save_at to a frozen copy of the model at that moment.
    live is "plot" (a picture that updates), "text" (one line that updates) or
    False. detail=True adds the training-number curves and the right-mod-d
    curves to the picture.
    This is the loop from notebooks/01_train.ipynb with a few extra options.
    Pressing stop ends training early and keeps the network as it is.
    """
    torch.manual_seed(seed)
    X, y = encode(data.numbers, data.n_digits), data.labels
    model = make_model(X.shape[1], width, data.modulus, layers)
    model.encode = encode
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history, saved = [], {}
    show = lambda done: progress(history, live, data.modulus, title, done, detail)
    try:
        for epoch in range(epochs + 1):
            history.append(evaluate(model, X, y, data, epoch))
            if epoch in save_at:
                saved[epoch] = copy.deepcopy(model)
            show(epoch == epochs)
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
        show(True)
        print(f"Stopped early at epoch {history[-1]['epoch']}.")
    return model, history, saved


def backup_train(encode, epochs=400, show="plot"):
    """The session 1 training loop, ready-made, for anyone whose own loop is stuck."""
    model, history, _ = train(encode, make_data(), epochs=epochs, live=show)
    return model, history


def check_loop(train_function):
    """Run a student's training loop for one epoch and check that the loss falls."""
    if not callable(train_function):
        print("Pass the loop itself: mlmath.check_loop(train). Run the cell with the loop first.")
        return
    try:
        returned = train_function(encode_onehot, epochs=1, show=None)
        if returned is None:
            print("Your loop gave back nothing. Keep its last line exactly as it was: return model, history")
            return
        _, history = returned
        history = _as_history(history)
        before, after = history[0]["train_loss"], history[-1]["train_loss"]
    except Exception as error:
        print("Your loop stopped with an error:")
        print("   ", f"{type(error).__name__}: {error}")
        print("Compare your two lines with Show answer under the loop, fix them, and run both cells again.")
        print("Mind the spelling, the dot, the brackets () at the end, and the indentation.")
        return
    if before - after > 0.05:
        print(f"Your loop works. One epoch of training lowered the loss from {before:.2f} to {after:.2f}.")
    else:
        print(f"Your loop runs, but the network is not learning: the loss stayed at {after:.2f}.")
        print("Check that you typed both lines, steps 3 and 4, each ending in (), "
              "lined up exactly under optimizer.zero_grad(). Then run both cells again.")


def summarize(history, factors=False):
    """One line: where the run ended."""
    last = history[-1]
    parts = [f"epoch {last['epoch']}", f"train accuracy {last['train_acc']:.1%}",
             f"test accuracy {last['test_acc']:.1%}"]
    if factors:
        parts += [f"right mod {k[3:-4]}: {v:.1%}" for k, v in last.items()
                  if k.startswith("mod") and _prime_power(int(k[3:-4]))]
    print(" | ".join(parts))


@_forgiving
def result(history, guess=Ellipsis, modulus=6):
    """How the run ended, next to the student's prediction."""
    last = _as_history(history)[-1]
    print(f"After {last['epoch']} epochs the network is right on {last['test_acc']:.0%} of the test numbers.")
    print(f"Guessing at random would be right {1 / modulus:.0%} of the time.")
    reached = first_epoch(_as_history(history))
    if reached is not None:
        print(f"It was first right on 90% of the test numbers at epoch {reached}.")
    if _blank(guess):
        print("You did not type a prediction. Next time, commit to a number before you run!")
        return
    try:
        value = _as_number(guess, "a percent")
    except Unreadable:
        print(f"Could not read your prediction {guess!r} as a number. Next time, type just the digits, like 50.")
        return
    print(f"You predicted {value:g}%.")
    if 0 < value < 1:
        print(f"(If you meant {value:.0%}, type {value * 100:g} next time: the prediction is a percent.)")
    elif not 0 <= value <= 100:
        print("(A percent of the test numbers is always between 0 and 100.)")


@_forgiving
def look_at(history, epochs):
    """Print loss and accuracy at a few epochs, with e^loss as 'answers still in play'."""
    history = _as_history(history)
    last = history[-1]["epoch"]
    good = []
    for epoch in _flatten(epochs):
        try:
            value = _as_int(epoch)
        except Unreadable:
            print(f"(skipping {epoch!r}: not a whole number)")
            continue
        if 0 <= value <= last:
            good.append(value)
        else:
            print(f"(skipping {value}: this run has epochs 0 to {last})")
    if not good:
        raise Unreadable(f"Put some whole numbers from 0 to {last} in the list, like [0, 30, {last}].")
    print("epoch    loss    answers still in play (e^loss)    exactly right")
    for epoch in sorted(set(good)):
        row = history[epoch]
        print(f"{epoch:5d}    {row['test_loss']:4.2f}    {math.exp(row['test_loss']):17.1f}"
              f"                 {row['test_acc']:9.0%}")

# ---------------------------------------------------------------- what the loss is, and what loss curves look like


def _read_beliefs(beliefs, modulus):
    """Beliefs as a list of `modulus` fractions adding to 1. Percents, fractions and {answer: belief} all work."""
    if isinstance(beliefs, dict):
        values = [0.0] * modulus
        for key, value in beliefs.items():
            answer = _as_int(key, "an answer")
            if not 0 <= answer < modulus:
                raise Unreadable(f"The answers go from 0 to {modulus - 1}, so {answer} can't be one of them.")
            values[answer] = _as_number(value, "a belief")
    else:
        items = _flatten(beliefs)
        if any(_blank(v) for v in items):
            raise Unreadable("Replace every ... with a belief (a percent, like 50) and run the cell again.")
        if len(items) != modulus:
            raise Unreadable(f"Type one belief for each of the {modulus} answers 0 to {modulus - 1}, "
                             f"so {modulus} numbers in the list. You typed {len(items)}.")
        values = [_as_number(v, "a belief") for v in items]
    if any(v < 0 for v in values):
        raise Unreadable("A belief can't be negative. Use 0 for an answer the network rules out.")
    total = sum(values)
    if total <= 0:
        raise Unreadable("At least one belief has to be bigger than 0.")
    if abs(total - 100) > 1.5 and abs(total - 1) > 0.015:
        print(f"(Your beliefs add up to {total:g}, not 100. They have been scaled so that they do.)")
    return [v / total for v in values]


@_forgiving
def loss_for(beliefs, answer, modulus=6):
    """The loss on one number: minus the natural log of the belief in the right answer."""
    p = _read_beliefs(beliefs, modulus)
    answer = _as_int(answer, "the right answer")
    if not 0 <= answer < modulus:
        raise Unreadable(f"The right answer is a remainder, so a whole number from 0 to {modulus - 1}.")
    if p[answer] == 0:
        summary = "loss = -ln(0) = infinity. The network said the right answer was impossible."
    else:
        loss = abs(math.log(p[answer]))  # abs() so a perfect answer prints 0.00, not -0.00
        summary = f"loss = -ln({p[answer]:.2f}) = {loss:.2f}" + ("  (perfect)" if loss < 0.005 else "")
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.bar(range(modulus), p, color=[BLUE if a == answer else LIGHT_BLUE for a in range(modulus)], width=0.65)
    for a, v in enumerate(p):
        ax.text(a, v + 0.03, "·" if v < 0.005 else f"{v:.0%}", ha="center", va="bottom", fontsize=9,
                fontweight="bold" if a == answer else "normal")
    ax.set_xticks(range(modulus))
    ax.set_xticklabels([f"{a}\n(right)" if a == answer else str(a) for a in range(modulus)])
    ax.set_xlabel("answer")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_ylabel("belief")
    ax.grid(axis="x", visible=False)
    ax.set_title(summary.split(". ")[0].replace("-ln", "−ln"), loc="left")
    fig.tight_layout()
    plt.show()
    print(f"The right answer is {answer}, and the network gives it {p[answer]:.0%}.")
    print(summary)


def first_epoch(history, level=0.9):
    """The first epoch with test accuracy at or above level, or None."""
    return next((row["epoch"] for row in history if row["test_acc"] >= level), None)


def _made_up_runs():
    """Four invented training runs, (training loss, test loss) for epochs 0 to 400."""
    e = np.arange(401.0)
    start = math.log(6)
    wiggle = lambda seed: 0.01 * np.random.default_rng(seed).standard_normal(len(e))
    sudden = start / (1 + np.exp((e - 90) / 6)) + 0.03
    stuck = start - 0.015 * np.log1p(e) + wiggle(1)
    memorize_train = start * np.exp(-e / 45) + 0.01
    memorize_test = start - 0.25 * (1 - np.exp(-e / 10)) + 1.1 * (1 - np.exp(-np.maximum(e - 15, 0) / 120))
    steady = start * np.exp(-e / 110) + 0.03
    return {"A": (sudden - 0.02, sudden), "B": (stuck - 0.03, stuck + wiggle(2)),
            "C": (memorize_train, memorize_test), "D": (steady - 0.02, steady)}


def loss_curve_examples():
    """Four made-up loss curves, A to D, each telling a different story."""
    runs = _made_up_runs()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    e = np.arange(401)
    for ax, (letter, (train_loss, test_loss)) in zip(axes.flat, runs.items()):
        ax.plot(e, train_loss, color=BLUE, alpha=0.45, linestyle="--", label="training numbers")
        ax.plot(e, test_loss, color=BLUE, label="test numbers")
        ax.axhline(math.log(6), color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(f"Curve {letter}", loc="left")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 3)
        ax.set_yticks([0, 1, math.log(6), 3])
        ax.set_yticklabels(["0", "1", "ln 6", "3"])
        ax.set_xticks([0, 1, 3, 10, 30, 100, 300])
        ax.set_xticklabels(["0", "1", "3", "10", "30", "100", "300"])
    axes[0, 0].legend(loc="lower left", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("epochs (stretched scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("loss")
    fig.suptitle("Four made-up training runs", x=0.06, ha="left", fontweight="bold")
    fig.tight_layout()
    plt.show()


CURVE_STORIES = {"never_learns": "B", "memorizes": "C", "learns_suddenly": "A", "learns_steadily": "D"}
_CURVE_TIPS = {
    "never_learns": "Look for a loss that stays near ln 6, where it started.",
    "memorizes": "Look for a curve where the two lines split apart: the training loss falls but the test loss doesn't.",
    "learns_suddenly": "Look for a long flat stretch followed by a cliff.",
    "learns_steadily": "Look for a loss that slides down a little at a time.",
}


def _read_letter(value):
    value = _unwrap(value)
    if _blank(value):
        return None
    letters = re.sub(r"(?i)curve", "", str(value)).strip(" .,'\"").upper()
    return letters if letters in ("A", "B", "C", "D") else str(value)


@_forgiving
def check_curves(never_learns, memorizes, learns_suddenly, learns_steadily):
    """Score a student's matching of stories to the made-up curves A to D."""
    typed = dict(never_learns=never_learns, memorizes=memorizes,
                 learns_suddenly=learns_suddenly, learns_steadily=learns_steadily)
    right = 0
    for story, value in typed.items():
        letter = _read_letter(value)
        name = story.replace("_", " ")
        if letter is None:
            print(f"   {name:16s} not filled in yet: replace the ... with a letter in quotes, like \"A\"")
        elif letter == CURVE_STORIES[story]:
            right += 1
            print(f"✓  {name:16s} curve {letter}")
        elif letter in ("A", "B", "C", "D"):
            print(f"✗  {name:16s} not curve {letter}. {_CURVE_TIPS[story]}")
        else:
            print(f"   {name:16s} could not read {letter!r}. Type one letter in quotes, like \"A\".")
    if right == 4:
        print("\nAll four right. Keep these pictures in mind when you sketch your own prediction later.")


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


def plot_history(history, modulus=6, title=None, marks=(), factors=False, train_curves=False):
    """Loss (top) and accuracy (bottom) against epochs, on a stretched epoch axis.

    The dotted lines in the loss panel sit at ln(k). A network that has narrowed
    every number down to k equally likely answers has loss exactly ln(k).
    marks is a list of epochs to flag with a vertical line. factors=True adds
    one accuracy curve per divisor d of the modulus (right mod d).
    train_curves=True adds dashed curves for the training numbers.
    """
    epochs = _column(history, "epoch")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    top.plot(epochs, _column(history, "test_loss"), color=BLUE, label="test numbers")
    if train_curves:
        top.plot(epochs, _column(history, "train_loss"), color=BLUE, alpha=0.45, linestyle="--",
                 label="training numbers")
    right = max(epochs[-1], 1)
    levels = sorted({modulus} | {modulus // d for d in divisors(modulus)})
    highest = max(max(row["test_loss"], row["train_loss"] if train_curves else 0) for row in history)
    for k in levels:
        top.axhline(math.log(k), color=MUTED, linewidth=1, linestyle=":")
    top.set_ylim(0, max(math.log(modulus) * 1.12, highest * 1.05))
    labeled = []  # label a dotted line only if it has room; the top one (ln modulus) always gets a label
    for k in reversed(levels):
        if not labeled or math.log(labeled[-1]) - math.log(k) >= 0.065 * top.get_ylim()[1]:
            labeled.append(k)
    top.set_yticks([0] + [math.log(k) for k in labeled])
    top.set_yticklabels(["0"] + [f"ln {k} = {math.log(k):.2f}" for k in labeled])
    top.grid(axis="y", visible=False)
    top.set_ylabel("loss")
    top.set_title(title or f"Learning n mod {modulus}", loc="left")

    if factors:  # drawn first and wide, so the exact curve stays visible on top where they overlap
        for color, d in zip(SERIES[1:], pieces(modulus)):
            name = "right mod 2 (parity)" if d == 2 else f"right mod {d}"
            bottom.plot(epochs, _column(history, f"mod{d}_acc"), color=color, label=name, linewidth=4, alpha=0.8)
    bottom.plot(epochs, _column(history, "test_acc"), color=BLUE, label=f"exactly right (mod {modulus})")
    if train_curves:
        bottom.plot(epochs, _column(history, "train_acc"), color=BLUE, alpha=0.45, linestyle="--",
                    label="exactly right, training numbers")
    bottom.axhline(1 / modulus, color=MUTED, linewidth=1, linestyle=":")
    bottom.set_ylabel("accuracy on test numbers")
    bottom.set_ylim(-0.03, 1.03)
    if 1 / modulus >= 0.08:
        bottom.set_yticks([0, 1 / modulus, 0.5, 1])
        bottom.set_yticklabels(["0%", f"guessing {1 / modulus:.0%}", "50%", "100%"])
    else:  # guessing sits too close to 0% for both labels
        bottom.set_yticks([1 / modulus, 0.5, 1])
        bottom.set_yticklabels([f"guessing {1 / modulus:.0%}", "50%", "100%"])
    bottom.set_xlabel("epochs of training (stretched scale)")
    if train_curves:
        top.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    if train_curves:
        bottom.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    elif factors:
        bottom.legend(loc="lower right", fontsize=9)
    for ax in (top, bottom):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, right)
        for m in marks:
            ax.axvline(m, color=INK, linewidth=1, alpha=0.5)
            if ax is top:
                ax.annotate(f"epoch {m}", (m, ax.get_ylim()[1]), xytext=(3, -12),
                            textcoords="offset points", fontsize=9, color=INK)
    ticks = [t for t in (0, 1, 3, 10, 30, 100, 300, 1000) if t <= right]
    bottom.set_xticks(ticks)
    bottom.set_xticklabels([str(t) for t in ticks])
    return fig


@_forgiving
def show_history(history, modulus=6, marks=()):
    """Draw a finished training run."""
    plot_history(_as_history(history), modulus, marks=marks)
    plt.show()


def show_reveal(history):
    """The session 1 run again, with the right-mod-2 and right-mod-3 curves drawn in."""
    fig = plot_history(history, 6, title="Learning n mod 6, with the hidden curves", factors=True)
    top = fig.axes[0]
    top.annotate("first drop:\nlearns n mod 2", (1.6, 1.45), xytext=(12, 8), textcoords="offset points",
                 fontsize=10, color=ORANGE, fontweight="bold")
    top.annotate("second drop:\nlearns n mod 3", (85, 0.75), xytext=(14, 4), textcoords="offset points",
                 fontsize=10, color=AQUA, fontweight="bold")
    plt.show()


LIVE_SECONDS = 1.5  # at most one redraw this often, so the picture never flashes
LIVE_DPI = 90
_live = {"handle": None, "last": 0.0, "image": None}


def _frame(history, modulus, title, detail):
    """The training plot as PNG bytes, always the same pixel size for a given run."""
    # detail=True adds the right-mod-d curves and the training curves; detail="train" only the training curves
    fig = plot_history(history, modulus, title, factors=detail is True, train_curves=bool(detail))
    size = (9.5, 6.4) if detail else (7.5, 6.4)  # room for the legend only when there is one
    fig.set_size_inches(*size)
    fig.subplots_adjust(left=0.19 if not detail else 0.15, right=0.71 if detail else 0.96,
                        top=0.94, bottom=0.09)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=LIVE_DPI)
    plt.close(fig)
    return buffer.getvalue(), size


def progress(history, show="plot", modulus=6, title=None, done=False, detail=False):
    """Show training progress, updated in place. Call it once per epoch.

    show="plot" draws the loss and accuracy picture, show="text" prints one line,
    and show=None shows nothing. Either way it redraws at most every
    LIVE_SECONDS seconds, always draws the first and last frame, and never
    clears the output, so nothing flashes.
    """
    if not show:
        return
    show = "plot" if show is True else str(show).lower()
    first = len(history) == 1
    now = time.time()
    if not (first or done or now - _live["last"] >= (LIVE_SECONDS if show == "plot" else 0.3)):
        return
    _live["last"] = now
    from IPython.display import HTML, display
    if show == "text":
        row = history[-1]
        text = (f"epoch {row['epoch']:4d}   loss {row['test_loss']:.2f}   "
                f"right on {row['test_acc']:4.0%} of test numbers" + ("   (done)" if done else ""))
        content = {"text/plain": text}
        if first or _live["handle"] is None:
            _live["handle"] = display(content, raw=True, display_id=True)
        else:
            _live["handle"].update(content, raw=True)
        return
    png, size = _frame(history, modulus, title, detail)
    if _live["image"] is not None:  # the explore panel supplies its own picture widget
        _live["image"].value = png
        return
    width, height = int(size[0] * LIVE_DPI), int(size[1] * LIVE_DPI)
    picture = HTML(f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}" '
                   f'width="{width}" height="{height}" style="max-width:100%;height:auto">')
    if first or _live["handle"] is None:
        _live["handle"] = display(picture, display_id=True)
    else:
        _live["handle"].update(picture)

# ---------------------------------------------------------------- probes

POSITION_NAMES = {4: ["thousands", "hundreds", "tens", "ones"]}


def position_name(position, n_digits=4):
    return POSITION_NAMES.get(n_digits, [f"position {i}" for i in range(n_digits)])[position]


def _position(place, n_digits=4):
    """A place as a name ("thousands", "Thousand", "hundreds place") or a position (0 = leftmost)."""
    names = POSITION_NAMES.get(n_digits, [])
    place = _unwrap(place)
    if isinstance(place, str):
        words = re.sub(r"[^a-z ]", " ", place.lower()).split()
        words = [w for w in words if w not in ("the", "place", "digit", "s")]
        stem = " ".join(words).rstrip("s")
        aliases = {"thousand": 0, "hundred": 1, "ten": 2, "one": 3, "unit": 3, "last": 3}
        if stem in aliases and n_digits == 4:
            return aliases[stem]
        raise Unreadable(f"Could not read the place {place!r}. Use one of: "
                         + ", ".join(f'"{n}"' for n in names) + ".")
    if isinstance(place, (int, np.integer)) and not isinstance(place, bool) and 0 <= place < n_digits:
        return int(place)
    raise Unreadable("The place should be one of: " + ", ".join(f'"{n}"' for n in names)
                     + ", in quotation marks.")


@torch.no_grad()
def predict_all(model, numbers, n_digits=4):
    """The network's answer for each number."""
    return model(model.encode(torch.as_tensor(numbers), n_digits)).argmax(1)


def _read_numbers(values, low, high):
    """Whole numbers from low to high; prints a note for anything skipped."""
    good = []
    for value in _flatten(values):
        try:
            n = _as_int(value)
        except Unreadable:
            print(f"(skipping {value!r}: type whole numbers, with no commas or periods inside a number)")
            continue
        if low <= n <= high:
            good.append(n)
        else:
            print(f"(skipping {n}: pick whole numbers from {low} to {high})")
    return good


@_forgiving
@torch.no_grad()
def ask(models, *numbers, n_digits=4):
    """Ask one or more networks about one or more numbers. models is a dict {name: model}.

    Prints how sure each network is of each possible answer. A dot means
    under 1%: the network has ruled that answer out.
    """
    models = _as_models(models)
    if not numbers or all(_blank(n) for n in numbers):
        raise Unreadable("Put some numbers in the list, like [4712, 2025].")
    numbers = _read_numbers(numbers, 0, 10 ** n_digits - 1)
    if not numbers:
        return
    width = max(len(name) for name in models) + 2
    for n in numbers:
        modulus = None
        rows = []
        for name, model in models.items():
            probs = torch.softmax(model(model.encode(torch.tensor([n]), n_digits)), dim=1)[0]
            modulus = len(probs)
            cells = "".join(f"{'·' if p < 0.005 else f'{p:.0%}':>6}" for p in probs.tolist())
            guess = int(probs.argmax())
            verdict = "right" if guess == n % modulus else "wrong"
            rows.append(f"  {name:<{width}}{cells}     says {guess} ({verdict})")
        print(f"{n:0{n_digits}d}   true answer: {n} mod {modulus} = {n % modulus}")
        print(f"  {'answer:':<{width}}" + "".join(f"{a:>6}" for a in range(modulus)))
        print("\n".join(rows) + "\n")
    print("Each row shows how sure the network is of each answer. A dot means under 1%: ruled out.")


@_forgiving
@torch.no_grad()
def right_mod(models, data, d):
    """For each network: how often its answer and the true answer agree mod d (test numbers)."""
    data = _as_data(data)
    models = _as_models(models)
    if _blank(_unwrap(d)):
        raise Unreadable("Replace the ... with a whole number, like 2, and run the cell again.")
    choices = ", ".join(map(str, divisors(data.modulus) + [data.modulus]))
    values = _read_numbers(d, 1, data.modulus)
    if not values:
        raise Unreadable(f"Type a whole number, like 2. Good ones to try: {choices}.")
    for d in values:
        if data.modulus % d:
            print(f"{d} does not divide {data.modulus}, so 'right mod {d}' does not mean much "
                  f"for n mod {data.modulus}. Try one of: {choices}.\n")
            continue
        truth = data.labels[data.test]
        # what blind guessing would score: a uniformly random answer, checked mod d
        guessing = np.mean([(a % d == truth % d).float().mean().item() for a in range(data.modulus)])
        print(f"How often the network's answer is right mod {d}, over {len(truth):,} test numbers")
        print(f"(right mod {d} means the answer and the truth leave the same remainder after dividing by {d})")
        for name, model in models.items():
            pred = predict_all(model, data.numbers[data.test], data.n_digits)
            print(f"   {name:10s}  {(pred % d == truth % d).float().mean().item():5.0%}")
        print(f"   {'guessing':10s}  {guessing:5.0%}\n")


@_forgiving
@torch.no_grad()
def mistakes(models, data, how_many=10):
    """A sample of test numbers each network gets wrong: the number, the true answer, and its answer."""
    data = _as_data(data)
    models = _as_models(models)
    how_many = _as_int(how_many, "how many mistakes to show")
    if not 1 <= how_many <= 100:
        raise Unreadable("Pick how many mistakes to show: a whole number from 1 to 100.")
    numbers, truth = data.numbers[data.test], data.labels[data.test]
    for name, model in models.items():
        pred = predict_all(model, numbers, data.n_digits)
        wrong = (pred != truth).nonzero().flatten()
        if len(wrong) == 0:
            print(f"{name}: right on every test number.\n".replace("this network:", "The network is"))
            continue
        generator = torch.Generator().manual_seed(0)  # the same sample every time
        sample = wrong[torch.randperm(len(wrong), generator=generator)[:how_many]]
        sample = sample[numbers[sample].argsort()]
        label = "The network is" if name == "this network" else f"{name}:"
        print(f"{label} wrong on {len(wrong):,} of {len(numbers):,} test numbers. Here are {len(sample)} of them.")
        print("    number    true answer    network says")
        for i in sample.tolist():
            print(f"    {int(numbers[i]):04d}    {int(truth[i]):>11d}    {int(pred[i]):>12d}")
        print()


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
            ax.text(j, i, fmt(table[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if table[i, j] > middle else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.grid(False)
    ax.tick_params(length=0)


@_forgiving
def plot_confusions(models, data):
    """One confusion table per model. models is a dict {name: model}."""
    data = _as_data(data)
    models = _as_models(models)
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


@torch.no_grad()
def swap_test(model, data, position, shift):
    """Add shift to one digit and see whether the network changes its answer.

    Uses every number whose digit in that position can take the shift
    without leaving 0..9. Returns the fraction of answers left unchanged.
    """
    position = _position(position, data.n_digits)
    numbers = data.numbers
    digit = digits_of(numbers, data.n_digits)[:, position]
    ok = (digit + shift >= 0) & (digit + shift <= 9)
    numbers = numbers[ok]
    swapped = numbers + shift * 10 ** (data.n_digits - 1 - position)
    same = predict_all(model, numbers, data.n_digits) == predict_all(model, swapped, data.n_digits)
    return same.float().mean().item()


@_forgiving
def change_digit(models, data, place, shifts):
    """Print how often each network keeps its answer when one digit goes up by each shift."""
    data = _as_data(data)
    models = _as_models(models)
    position = _position(place, data.n_digits)
    shifts = list(dict.fromkeys(s for s in _read_numbers(shifts, -9, 9) if s != 0))  # no repeats, no 0
    if not shifts:
        raise Unreadable("Put some whole numbers from 1 to 9 in the list, like [1, 2, 3].")
    value = 10 ** (data.n_digits - 1 - position)
    names = list(models)
    print(f"How often the answer stays the same when we add s to the {position_name(position, data.n_digits)} digit")
    print("   add s   " + "".join(f"{name:>12}" for name in names) + "     the true answer")
    for s in shifts:
        cells = "".join(f"{swap_test(models[name], data, position, s):12.0%}" for name in names)
        stays = (s * value) % data.modulus == 0
        print(f"   {s:+4d}    {cells}     {'stays' if stays else 'changes'}"
              f"  ({s} x {value} mod {data.modulus} = {(s * value) % data.modulus})")


@torch.no_grad()
def swap_table(model, data, position):
    """table[a, b] = how often the answer survives replacing digit a by digit b."""
    position = _position(position, data.n_digits)
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


@_forgiving
def plot_swap_tables(models, data, place):
    """Swap tables for one or more networks, side by side. models is {name: model}."""
    data = _as_data(data)
    models = _as_models(models)
    position = _position(place, data.n_digits)
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 4.9), squeeze=False)
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


GROUPS_EXAMPLE = "[[0, 1], [2, 3, 4], [5, 6, 7, 8, 9]]"


def _read_group(group):
    """One group of digits: [0, 3, 6], (0, 3, 6), {0, 3, 6}, "0 3 6", "036" or 369 all work."""
    if isinstance(group, str):
        return [int(c) for c in group if c.isdigit()]
    if isinstance(group, (list, tuple, set, frozenset, range)):
        digits = []
        for item in _flatten(group):
            if isinstance(item, str):
                digits += [int(c) for c in item if c.isdigit()]
            else:
                n = _as_int(item)
                if n < 0:
                    raise Unreadable(f"{n} is not a digit. Use only the digits 0 to 9.")
                digits += [int(c) for c in str(n)]  # 369 typed without commas means 3, 6, 9
        return digits
    n = _as_int(group)
    if n < 0:
        raise Unreadable(f"{n} is not a digit. Use only the digits 0 to 9.")
    return [int(c) for c in str(n)]


def _read_groups(groups):
    """A grouping of the digits 0 to 9, from a list of lists or a string like "0369 147 258"."""
    groups = _unwrap(groups) if not isinstance(groups, list) else groups
    if _blank(groups):
        raise Unreadable(f"Replace the ... with your groups, written like this: {GROUPS_EXAMPLE}")
    if isinstance(groups, str):
        parts = re.split(r"[;|/]", groups) if re.search(r"[;|/]", groups) else re.split(r"[\s,]+", groups)
        return [g for g in (_read_group(p) for p in parts) if g]
    if isinstance(groups, dict):
        groups = list(groups.values())
    if not isinstance(groups, (list, tuple, set)):
        raise Unreadable(f"Type your groups as a list of lists, like this: {GROUPS_EXAMPLE}")
    groups = list(groups)
    if groups and all(isinstance(g, (int, np.integer)) and 0 <= g <= 9 for g in groups):
        raise Unreadable("This looks like one flat list of digits. Put each group in its own square "
                         f"brackets, inside one big pair of brackets, like this: {GROUPS_EXAMPLE}")
    return [g for g in (_read_group(g) for g in groups) if g]


@_forgiving
def check_groups(model, data, groups, place):
    """Score a student's grouping of the digits 0-9 against the network's swap table."""
    data = _as_data(data)
    model = list(_as_models(model).values())[0]
    groups = _read_groups(groups)
    flat = [d for g in groups for d in g]
    missing = sorted(set(range(10)) - set(flat))
    repeated = sorted({d for d in flat if flat.count(d) > 1})
    if missing or repeated:
        if missing:
            print(f"Every digit should be in exactly one group. Missing: {missing}. "
                  "(A digit on its own is a group of one, like [4].)")
        if repeated:
            print(f"These digits are in more than one group: {repeated}.")
        return
    position = _position(place, data.n_digits)
    table = swap_table(model, data, position)
    group_of = {d: i for i, g in enumerate(groups) for d in g}
    same = [table[a, b].item() for a in range(10) for b in range(10) if a != b and group_of[a] == group_of[b]]
    other = [table[a, b].item() for a in range(10) for b in range(10) if group_of[a] != group_of[b]]
    shown = "  ".join("{" + ", ".join(map(str, sorted(g))) + "}" for g in groups)
    print(f"Your groups for the {position_name(position, data.n_digits)} digit:  {shown}")
    if same:
        print(f"   swap two digits from the same group:       answer stays the same {np.mean(same):4.0%} of the time")
    if other:
        print(f"   swap two digits from different groups:     answer stays the same {np.mean(other):4.0%} of the time")
    print("A perfect grouping scores 100% and 0%.")
    if same and other and np.mean(same) > 0.9 and np.mean(other) < 0.05:
        print("These are the groups the network uses.")


@_forgiving
def check_recipe(recipe, model, data):
    """Run a student's recipe on every number and compare it with n mod 6 and with the network."""
    data = _as_data(data)
    if not callable(recipe):
        raise Unreadable("Pass the recipe itself: mlmath.check_recipe(recipe, ...). "
                         "Run the cell that starts with def recipe(n): first.")
    model = list(_as_models(model).values())[0]
    answers = []
    for n in data.numbers.tolist():
        try:
            answer = recipe(n)
        except IndexError:
            print(f"For n = {n}, no number from 0 to 5 matched your parity and r3.")
            print("parity should come out as 0 or 1 (use % 2), and r3 as 0, 1 or 2 (use % 3).")
            print("Did you fill in both blanks? Compare with Show answer under the cell.")
            return
        except Exception as error:
            print(f"Your recipe stopped with an error at n = {n}:  {type(error).__name__}: {error}")
            print("Did you fill in both blanks? Compare with Show answer under the cell.")
            return
        try:
            answers.append(_as_int(answer))
        except Unreadable:
            if answer is None:
                print("Your recipe gave back nothing. Keep the last line, the one that starts with return.")
                return
            answers.append(-1)
    answers = torch.tensor(answers)
    total = len(answers)
    network = predict_all(model, data.numbers, data.n_digits)
    print(f"Your recipe agrees with n mod {data.modulus} on {(answers == data.labels).sum().item():,} "
          f"of the {total:,} numbers.")
    print(f"It agrees with the trained network on {(answers == network).sum().item():,} of them.")

# ---------------------------------------------------------------- pictures of networks



def _node(ax, x, y, r, fill="white", edge=EDGE, lw=1.2, z=3, dashed=False):
    ax.add_patch(plt.Circle((x, y), r, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z,
                            linestyle=(0, (2, 1.5)) if dashed else "-"))


def _dots(ax, x, y, gap=0.13):
    """Three small dots in a column: this layer has more units than are drawn."""
    ax.plot([x, x, x], [y - gap, y, y + gap], "o", color=MUTED, markersize=2.2, zorder=3)


def _slots(size, shown):
    """Which units of a layer to draw: all of them, or the first and last few with a gap (None) between."""
    if size <= shown:
        return list(range(size))
    half = shown // 2
    return list(range(half)) + [None] + list(range(size - half, size))


def _spread(count, spacing):
    """count heights, spacing apart, centered on 0, from the top down."""
    return [(count - 1) / 2 * spacing - j * spacing for j in range(count)]


# A tiny network with its connection strengths set by hand. Its job: is x between 0.35 and 0.65?
TINY = {"w_in": np.array([10.0, 10, 10, 10, -10]), "b_in": np.array([-3.0, -4, -6, -7, 2]),
        "w_out": np.array([10.0, -10, -10, 10, -2]), "b_out": -5.0}


def _tiny(x):
    """Hidden values, output score and belief in "yes" of the tiny network at input x."""
    hidden = np.maximum(0, np.multiply.outer(np.asarray(x, dtype=float), TINY["w_in"]) + TINY["b_in"])
    score = hidden @ TINY["w_out"] + TINY["b_out"]
    return hidden, score, 1 / (1 + np.exp(-score))


@_forgiving
def draw_tiny_network(x=None):
    """A network with 1 input, 5 hidden units and 1 output, and what it computes.

    With x, every unit shows the value it holds when the input is x.
    """
    if x is not None:
        x = _as_number(x, "a number from 0 to 1 (like 0.5)")
        if not 0 <= x <= 1:
            raise Unreadable(f"{x:g} is outside the picture. Pick a number from 0 to 1, like 0.5.")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    left.set_aspect("equal")
    left.axis("off")
    hidden_y = _spread(5, 0.95)
    r = 0.33
    if x is not None:
        hidden, score, belief = _tiny(x)
    for i, y in enumerate(hidden_y):
        for (x1, y1, x2, y2, w, faded) in [(0, 0, 2.6, y, TINY["w_in"][i], False),
                                           (2.6, y, 5.2, 0, TINY["w_out"][i], x is not None and hidden[i] == 0)]:
            left.plot([x1, x2], [y1, y2], color=BLUE if w > 0 else ORANGE, linewidth=0.8 + 2.2 * abs(w) / 10,
                      alpha=0.12 if faded else 0.75, zorder=1, solid_capstyle="round")
    if x is None:
        _node(left, 0, 0, r)
        for y in hidden_y:
            _node(left, 2.6, y, r)
        _node(left, 5.2, 0, r)
    else:
        _node(left, 0, 0, r, fill=LIGHT_BLUE)
        left.text(0, 0, f"{x:.2f}", ha="center", va="center", fontsize=9, zorder=4)
        for h, y in zip(hidden, hidden_y):
            _node(left, 2.6, y, r, fill=LIGHT_BLUE if h > 0 else OFF, edge=EDGE if h > 0 else MUTED)
            left.text(2.6, y, f"{h:.1f}", ha="center", va="center", fontsize=9, zorder=4,
                      color=INK if h > 0 else MUTED)
        _node(left, 5.2, 0, r, fill=SEQUENTIAL(0.15 + 0.7 * belief))
        left.text(5.2, 0, f"{belief:.0%}", ha="center", va="center", fontsize=9, zorder=4,
                  color="white" if belief > 0.6 else INK)
    left.text(0, -0.6, "input\nx", ha="center", va="top", fontsize=10)
    left.text(2.6, hidden_y[0] + 0.55, "5 hidden units", ha="center", va="bottom", fontsize=10)
    left.text(5.2, -0.6, "output\nbelief in \"yes\"", ha="center", va="top", fontsize=10)
    left.text(2.6, hidden_y[-1] - 0.7, "Each line is a connection strength (a number).\n"
              "Blue: positive.  Orange: negative.  Thicker: bigger.",
              ha="center", va="top", fontsize=9, color=MUTED)
    left.set_xlim(-0.9, 6.3)
    left.set_ylim(hidden_y[-1] - 1.6, hidden_y[0] + 0.9)
    left.set_title("The network" + ("" if x is None else f", given x = {x:.2f}"), loc="left")

    xs = np.linspace(0, 1, 401)
    right.axvspan(0.35, 0.65, color=AQUA, alpha=0.12, linewidth=0)
    right.text(0.5, 1.07, "the \"yes\" numbers", ha="center", va="bottom", fontsize=9, color=AQUA)
    right.plot(xs, _tiny(xs)[2], color=BLUE)
    right.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    if x is not None:
        right.plot([x], [belief], "o", color=INK, markersize=7, zorder=5)
        right.axvline(x, color=INK, linewidth=1, alpha=0.4)
    right.set_xlim(0, 1)
    right.set_ylim(-0.03, 1.2)
    right.set_yticks([0, 0.5, 1])
    right.set_yticklabels(["0%", "50%", "100%"])
    right.set_xlabel("input x")
    right.set_ylabel("belief in \"yes\"")
    right.set_title("What it computes", loc="left")
    fig.suptitle("A tiny network: is x between 0.35 and 0.65?", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    plt.show()


def _input_layout(n_digits=4, spacing=0.32, group_gap=0.5):
    """Heights of the one-hot switches, place by place, top to bottom: {(place, digit): y}."""
    rows = []
    y = 0.0
    for place in range(n_digits):
        for digit in range(10):
            rows.append(((place, digit), y))
            y -= spacing
        y -= group_gap
    middle = (rows[0][1] + rows[-1][1]) / 2
    return {key: y - middle for key, y in rows}


def _draw_one_network(ax, model, number, title, hide=None, shown=16):
    """The one-hot network: 40 switches in, a sample of hidden units, 6 answers out with their beliefs."""
    n_digits = 4
    inputs = _input_layout(n_digits)
    digits = digits_of(torch.tensor([number]), n_digits)[0].tolist()
    lit = {(place, d) for place, d in enumerate(digits)}
    first = model[0]
    width = first.out_features
    slots = _slots(width, shown)
    hidden_y = _spread(len(slots), 0.72)
    with torch.no_grad():
        x = model.encode(torch.tensor([number]), n_digits)
        activation = model[1](first(x))[0]
        beliefs = torch.softmax(model(x), dim=1)[0].tolist()
    modulus = len(beliefs)
    gap = min(1.5, 13.0 / modulus)
    out_y = _spread(modulus, gap)
    r_out, bar = min(0.3, 0.42 * gap), min(0.26, 0.7 * gap)
    xi, xh, xo = 0.0, 4.0, 8.0
    drawn = [unit for unit in slots if unit is not None]
    top = max(activation[drawn].max().item(), 1e-6)

    for key, y in inputs.items():
        is_hidden = hide is not None and key == hide
        for unit, hy in zip(slots, hidden_y):
            if unit is None:
                continue
            if is_hidden:
                ax.plot([xi, xh], [y, hy], color=ORANGE, linewidth=0.7, alpha=0.55, linestyle=(0, (3, 2)), zorder=2)
            elif key in lit:
                ax.plot([xi, xh], [y, hy], color=BLUE, linewidth=0.8, alpha=0.4, zorder=2)
            else:
                ax.plot([xi, xh], [y, hy], color=MUTED, linewidth=0.4, alpha=0.07, zorder=1)
    for unit, hy in zip(slots, hidden_y):
        if unit is None:
            continue
        for oy in out_y:
            ax.plot([xh, xo], [hy, oy], color=MUTED, linewidth=0.5, alpha=0.22, zorder=1)

    names = POSITION_NAMES[n_digits]
    for (place, d), y in inputs.items():
        on = (place, d) in lit
        if hide is not None and (place, d) == hide:
            _node(ax, xi, y, 0.12, fill=YELLOW if on else "white", edge=ORANGE, lw=1.4, dashed=True)
        else:
            _node(ax, xi, y, 0.12, fill=BLUE if on else "white", edge=EDGE if on else MUTED, lw=1)
        ax.text(xi - 0.25, y, str(d), ha="right", va="center", fontsize=6.5,
                color=INK if on else MUTED, fontweight="bold" if on else "normal")
        if d == 0:
            ax.text(xi - 0.75, y - 4.5 * 0.32, f"{names[place]}\n{digits[place]}", ha="right", va="center",
                    fontsize=9, color=INK)
    for unit, hy in zip(slots, hidden_y):
        if unit is None:
            _dots(ax, xh, hy, gap=0.16)
        else:
            level = activation[unit].item() / top
            _node(ax, xh, hy, 0.17, fill=SEQUENTIAL(0.05 + 0.6 * level) if level > 0 else OFF,
                  edge=EDGE if level > 0 else MUTED)
    best = int(np.argmax(beliefs))
    for a, (p, oy) in enumerate(zip(beliefs, out_y)):
        small = modulus > 12
        _node(ax, xo, oy, r_out, fill=SEQUENTIAL(0.1 + 0.8 * p))
        ax.text(xo, oy, str(a), ha="center", va="center", fontsize=6 if small else 10, zorder=4,
                color="white" if p > 0.5 else INK, fontweight="bold")
        ax.add_patch(plt.Rectangle((xo + 0.45, oy - bar / 2), 2.2 * p, bar, color=BLUE if a == best else LIGHT_BLUE,
                                   zorder=3, linewidth=0))
        note = "  (right answer)" if a == number % modulus else ""
        if note or not small or p >= 0.01:
            ax.text(xo + 0.55 + 2.2 * p, oy, f"{p:.0%}{note}", ha="left", va="center", fontsize=7 if small else 9,
                    fontweight="bold" if note else "normal")
    head = max(inputs.values()) + 0.6
    ax.text(xi, head, f"{len(inputs)} switches in", ha="center", va="bottom", fontsize=10)
    ax.text(xh, head, f"{width} hidden units", ha="center", va="bottom", fontsize=10)
    ax.text(xo, head, f"{modulus} answers out", ha="center", va="bottom", fontsize=10)
    ax.text(xo + 1.2, out_y[-1] - 0.9, "beliefs", ha="center", va="top", fontsize=9, color=MUTED)
    ax.set_xlim(-3.2, 12.3)
    ax.set_ylim(min(inputs.values()) - 0.8, head + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left")


@_forgiving
def draw_network(number=4712, models="before training", hide_place=None, hide_digit=None):
    """Draw the one-hot network, shown one number: the switches that are on, and its belief in each answer.

    models is a network, a dict {name: network}, or "before training" for a freshly made network.
    Only 16 of the hidden units are drawn. A hidden unit's shade shows how strongly it fires.
    hide_place and hide_digit mark one switch with a dashed outline: the one that was never on in training.
    """
    number = _as_int(number, "a whole number from 0 to 9999")
    if not 0 <= number <= 9999:
        raise Unreadable(f"{number} is out of range. Pick a whole number from 0 to 9999.")
    if isinstance(models, str) and models == "before training":
        torch.manual_seed(0)
        untrained = make_model(40, 512, 6)
        untrained.encode = encode_onehot
        models = {"Before training": untrained}
    models = _as_models(models)
    if any(getattr(m, "encode", None) is not encode_onehot or not isinstance(m[0], nn.Linear)
           or m[0].in_features != 40 for m in models.values()):
        raise Unreadable("This picture only works for a network that reads one-hot digits.")
    if any(len(m) != 3 for m in models.values()):
        raise Unreadable("This picture only draws networks with one hidden layer.")
    hide = None
    if hide_place is not None:
        hide = (_position(hide_place), _as_int(hide_digit, "the hidden digit"))
    fig, axes = plt.subplots(1, len(models), figsize=(6.6 * len(models), 7.6), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        title = f"{name}, shown {number:04d}" if name != "this network" else f"Shown {number:04d}"
        _draw_one_network(ax, model, number, title, hide=hide)
    if hide is not None:
        fig.text(0.02, 0.01, f"Dashed orange: the switch for a {hide[1]} in the {position_name(hide[0])} place, "
                 "which was never on during training.", fontsize=9, color=ORANGE)
    fig.tight_layout()
    plt.show()


def _draw_layers(ax, sizes, shown=(8, 22, 6), spacing=0.3, step=2.0):
    """A plain network: one column of units per layer, every unit joined to every unit in the next layer."""
    columns = []
    for i, size in enumerate(sizes):
        limit = shown[0] if i == 0 else shown[2] if i == len(sizes) - 1 else shown[1]
        slots = _slots(size, limit)
        columns.append([(i * step, y, s) for y, s in zip(_spread(len(slots), spacing), slots)])
    for left, right in zip(columns, columns[1:]):
        for x1, y1, s1 in left:
            for x2, y2, s2 in right:
                if s1 is not None and s2 is not None:
                    ax.plot([x1, x2], [y1, y2], color=MUTED, linewidth=0.5, alpha=0.3, zorder=1)
    for column in columns:
        for x, y, s in column:
            if s is None:
                _dots(ax, x, y)
            else:
                _node(ax, x, y, 0.12, fill="white" if x in (0, step * (len(sizes) - 1)) else LIGHT_BLUE)
    return columns


@_forgiving
def draw_network_shapes(shapes=None):
    """Networks of different sizes and shapes, side by side. shapes is a dict {name: [40, ..., 6]}."""
    if shapes is None:
        shapes = {"one layer of 512\n(session 1)": [40, 512, 6], "one layer of 16": [40, 16, 6],
                  "one layer of 2": [40, 2, 6], "two layers of 16": [40, 16, 16, 6]}
    fig, axes = plt.subplots(1, len(shapes), figsize=(2.75 * len(shapes), 4.4))
    for ax, (name, sizes) in zip(np.atleast_1d(axes), shapes.items()):
        columns = _draw_layers(ax, sizes)
        low = min(y for column in columns for _, y, _ in column)
        for column, size in zip(columns, sizes):
            ax.text(column[0][0], low - 0.35, str(size), ha="center", va="top", fontsize=9, color=MUTED)
        ax.set_title(name, fontsize=11)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-3.9, 3.4)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.01, 0.02, "Each column is a layer. The grey numbers count its units: 40 switches in, "
             "6 answers out, hidden units in between. The dots mean more units than are drawn.",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()


def draw_training_loop(batch=64, train_size=5000):
    """The four steps of training, as a loop."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3.1, 3.1)
    ax.axis("off")
    steps = [((0, 2.2), f"1. Show the network a batch\nof {batch} training numbers", False),
             ((3.1, 0), "2. Measure the loss:\nthe average of −ln p", False),
             ((0, -2.2), "3. Work out which way to nudge\neach connection strength\nloss.backward()", True),
             ((-3.1, 0), "4. Nudge them\noptimizer.step()", True)]
    for (x, y), text, typed in steps:
        ax.text(x, y, text, ha="center", va="center", fontsize=10.5, zorder=3, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#fdf3d8" if typed else "white",
                          edgecolor=YELLOW if typed else EDGE, linewidth=1.4))
    arrows = [((1.9, 2.0), (3.0, 0.75)), ((3.0, -0.75), (2.0, -1.8)),
              ((-2.0, -1.8), (-3.0, -0.6)), ((-3.0, 0.6), (-1.9, 2.0))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.6, mutation_scale=16,
                                    connectionstyle="arc3,rad=-0.25"))
    batches = math.ceil(train_size / batch)
    ax.text(0, 0, f"Go around {batches} times\n(all {train_size:,} training numbers)\n= one epoch",
            ha="center", va="center", fontsize=10, color=MUTED, linespacing=1.4)
    ax.text(5, -3.05, "Yellow: the two steps you type.", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_title("The training loop", loc="left")
    fig.tight_layout()
    plt.show()


@_forgiving
@torch.no_grad()
def compare_answers(models, start=4700, count=30):
    """For the numbers start, start + 1, ...: the true answer next to what each network says."""
    if isinstance(models, dict):
        missing = [name for name, model in models.items() if model is None]
        for name in missing:
            print(f"{name}: there's no network yet (its training cell didn't finish), so it's left out.")
        models = {name: model for name, model in models.items() if model is not None}
        if not models:
            return
    models = _as_models(models)
    start = _as_int(start, "the first number to show")
    count = _as_int(count, "how many numbers to show")
    if not 5 <= count <= 100:
        raise Unreadable("Show between 5 and 100 numbers.")
    if not 0 <= start <= 10000 - count:
        raise Unreadable(f"Pick a first number from 0 to {10000 - count}.")
    numbers = torch.arange(start, start + count)
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 2.5 * len(models) + 0.6), sharex=True, squeeze=False)
    for ax, (name, model) in zip(axes[:, 0], models.items()):
        modulus = model[-1].out_features
        truth = (numbers % modulus).numpy()
        said = predict_all(model, numbers).numpy()
        ax.plot(numbers, truth, "o", markersize=11, markerfacecolor="none", markeredgecolor=MUTED,
                markeredgewidth=1.5, label=f"true answer, n mod {modulus}")
        ax.plot(numbers, said, "o", markersize=5.5, color=BLUE, label="the network says")
        right = int((truth == said).sum())
        label = "The network" if name == "this network" else name
        ax.set_title(f"{label}: right on {right} of {count}", loc="left")
        ax.set_ylim(-0.6, modulus - 0.4)
        ax.set_yticks(range(modulus))
        ax.set_ylabel("answer")
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.15), ncol=2, fontsize=9)
    axes[-1, 0].set_xlabel("n")
    axes[-1, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    plt.show()


@_forgiving
def draw_pairs(modulus=6, m=2, k=3):
    """Put each number from 0 to modulus - 1 in the box for its pair (n mod m, n mod k)."""
    modulus = _as_int(modulus, "the modulus")
    m = _as_int(m, "the first factor")
    k = _as_int(k, "the second factor")
    if not (2 <= m <= 12 and 2 <= k <= 12 and 2 <= modulus <= 60):
        raise Unreadable("Keep the numbers small: a modulus up to 60, and m and k from 2 to 12.")
    boxes = {(i, j): [a for a in range(modulus) if a % m == i and a % k == j] for i in range(m) for j in range(k)}
    fig, ax = plt.subplots(figsize=(max(1.35 * k + 2.4, 6.5), 0.9 * m + 1.7))
    for (i, j), members in boxes.items():
        fill = OFF if not members else "#d6e6fa" if len(members) == 1 else "#f9d3c2"
        ax.add_patch(plt.Rectangle((j, -i - 1), 1, 1, facecolor=fill, edgecolor="white", linewidth=3))
        ax.text(j + 0.5, -i - 0.5, ", ".join(map(str, members)) if members else "empty", ha="center",
                va="center", fontsize=13 if members else 9, color=INK if members else MUTED,
                fontweight="bold" if members else "normal")
    for j in range(k):
        ax.text(j + 0.5, 0.12, f"n mod {k} = {j}", ha="center", va="bottom", fontsize=9)
    for i in range(m):
        ax.text(-0.1, -i - 0.5, f"n mod {m} = {i}", ha="right", va="center", fontsize=9)
    ax.set_xlim(-1.4, k + 0.05)
    ax.set_ylim(-m - 0.05, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"The numbers 0 to {modulus - 1}, each in the box\nfor its remainders mod {m} and mod {k}",
                 loc="left", fontsize=11)
    fig.tight_layout()
    plt.show()
    crowded = sum(len(v) > 1 for v in boxes.values())
    empty = sum(not v for v in boxes.values())
    if crowded or empty:
        print(f"{crowded} boxes hold more than one number and {empty} are empty.")
    else:
        print(f"Every box holds exactly one number: the pair pins down n mod {modulus}.")


@_forgiving
def draw_recipe(number=4712):
    """The network's recipe for n mod 6, worked out for one number."""
    n = _as_int(number, "a whole number from 0 to 9999")
    if not 0 <= n <= 9999:
        raise Unreadable(f"{n} is out of range. Pick a whole number from 0 to 9999.")
    digits = [int(c) for c in f"{n:04d}"]
    parity, r3 = digits[-1] % 2, sum(digits) % 3
    answer = [a for a in range(6) if a % 2 == parity and a % 3 == r3][0]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(-0.6, 11.4)
    ax.set_ylim(-2.6, 2.6)
    ax.axis("off")
    ys = _spread(4, 1.15)
    box = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=EDGE, linewidth=1.2)
    for place, (d, y) in enumerate(zip(digits, ys)):
        ax.text(0.9, y, f"{POSITION_NAMES[4][place]}: {d}", ha="center", va="center", fontsize=11, bbox=box, zorder=3)
    sum_text = " + ".join(map(str, digits))
    two = (4.9, 1.4)
    three = (4.9, -1.2)
    ax.text(*two, f"learned first\nlast digit mod 2\n{digits[-1]} mod 2 = {parity}", ha="center", va="center",
            fontsize=10.5, linespacing=1.4, zorder=3, bbox={**box, "edgecolor": ORANGE, "linewidth": 2})
    ax.text(*three, f"learned second\ndigit sum mod 3\n{sum_text} = {sum(digits)}\n{sum(digits)} mod 3 = {r3}",
            ha="center", va="center", fontsize=10.5, linespacing=1.4, zorder=3,
            bbox={**box, "edgecolor": AQUA, "linewidth": 2})
    ax.text(9.4, 0.1, f"the one answer from 0 to 5\nwith n mod 2 = {parity}\nand n mod 3 = {r3}\n\n"
            f"n mod 6 = {answer}", ha="center", va="center", fontsize=11, linespacing=1.4, zorder=3,
            bbox={**box, "facecolor": "#d6e6fa", "edgecolor": BLUE, "linewidth": 2})
    arrow = dict(arrowstyle="-|>", linewidth=1.5, mutation_scale=14, shrinkA=4, shrinkB=4)
    ax.annotate("", xy=(3.95, two[1]), xytext=(1.75, ys[-1]), arrowprops={**arrow, "color": ORANGE})
    for y in ys:
        ax.annotate("", xy=(3.75, three[1]), xytext=(1.75, y), arrowprops={**arrow, "color": AQUA, "alpha": 0.8})
    for start in (two, three):
        ax.annotate("", xy=(7.85, 0.1), xytext=(6.05, start[1]), arrowprops={**arrow, "color": BLUE})
    ax.set_title(f"The network's recipe, worked out for n = {n:04d}", loc="left")
    fig.tight_layout()
    plt.show()
    print(f"Check: {n} mod 6 = {n % 6}.")


# ---------------------------------------------------------------- hints under errors

_SHOW_ANSWER = "Compare the cell with <b>Show answer</b> under it."


def _syntax_hint(value):
    """A plain-English reading of a SyntaxError, from its message and the offending line."""
    message = str(getattr(value, "msg", value))
    line = (getattr(value, "text", "") or "").split("#")[0].strip()  # the offending line, comment removed
    if "leading zeros" in message:
        return "Write numbers without zeros in front: <code>7</code>, not <code>0007</code>."
    if re.search(r"\bmod\b", line):
        return "Python writes mod as <code>%</code>: type <code>n % 6</code>, not <code>n mod 6</code>."
    if re.search(r"\d\s*%\s*($|#|,|\]|\))", line) or re.search(r"(^|[=(\[,])\s*%\s*\d", line):
        return "Leave out the percent sign: type <code>50</code>, not <code>50%</code>."
    if isinstance(value, IndentationError) or "indent" in message:
        return ("The spaces at the start of a line are off. Each line must line up with the line above it. "
                "In the training loop, your two lines start exactly below <code>optimizer.zero_grad()</code>. "
                "Delete any spaces you added at the start of other lines.")
    if "positional argument follows keyword argument" in message:
        return ("A list needs square brackets around it: <code>epochs=[0, 30, 400]</code>, "
                "not <code>epochs=0, 30, 400</code>.")
    if "never closed" in message or "unmatched" in message or "does not match opening" in message:
        return ("A bracket is missing or extra. Every <code>(</code> needs a <code>)</code>, "
                "and every <code>[</code> needs a <code>]</code>.")
    if "forgot a comma" in message:
        return "Python expected a comma here. Separate numbers in a list with commas: <code>[1, 2, 3]</code>."
    if "unterminated string" in message or "EOL while scanning" in message:
        return "A quotation mark is missing. Text in quotes needs one at each end: <code>\"thousands\"</code>."
    if "expected ':'" in message or (re.match(r"(for|if|def|while|else|elif)\b", line) and not line.endswith(":")):
        return "A colon is missing at the end of the line: <code>for a in range(6):</code>"
    if "invalid decimal literal" in message or "invalid digit" in message:
        return "A number runs into letters or other symbols. Type plain whole numbers like <code>4712</code>."
    if "cannot assign" in message or "Maybe you meant '=='" in message:
        return "Keep the name on the left of the <code>=</code> as it was, and type only on the right."
    if "invalid character" in message:
        return ("There is a character Python does not understand, often a curly quote or a dash copied from "
                "a document. Retype it on the keyboard.")
    return ("Python could not read this cell. Look near the <code>^</code> mark for a typo: a missing comma or "
            "bracket, a stray period, or a word where a number should be.")


def _hint(etype, value, namespace):
    """What an error probably means, in plain words, as HTML. None if there is nothing to add."""
    message = str(value)
    if issubclass(etype, KeyboardInterrupt):
        return ("You stopped this cell, so it did not finish. Anything it would have made, like "
                "<code>history</code> after a training run, does not exist yet. Run it again and let it finish, "
                "or open <b>Show answer</b> under the cell to load the saved run instead.")
    if issubclass(etype, SyntaxError):
        return f"{_syntax_hint(value)} {_SHOW_ANSWER}"
    if etype.__name__ == "UsageError":
        return f"Leave out the percent sign: type <code>50</code>, not <code>%50</code> or <code>50%</code>. {_SHOW_ANSWER}"
    if issubclass(etype, NameError):
        name = getattr(value, "name", None) or (re.findall(r"'(\w+)'", message) or [""])[0]
        known = [k for k in namespace if not k.startswith("_")] + dir(builtins)
        tips = []
        if re.fullmatch(r"[A-Da-d]", name or ""):
            return (f"Put the letter in quotation marks: <code>\"{html.escape(name.upper())}\"</code>, "
                    f"not <code>{html.escape(name)}</code>. Without them, Python looks for something named "
                    f"{html.escape(name)}.")
        if re.match(r"(model|history)", name or "") and "train(" not in (namespace.get("_ih") or [""])[-1]:
            return (f"<code>{html.escape(name)}</code> is made by an earlier training cell, which hasn't run yet "
                    "or didn't finish. Scroll up to the training cell, run it and let it finish, then run this "
                    "cell again. If it won't finish, <b>Show answer</b> under the training cell shows how to load "
                    "the saved run instead.")
        if re.match(r"(model|history)", name or ""):
            return (f"<code>{html.escape(name)}</code> is made by a training line, and it does not exist yet. "
                    "Type the training line where the ✏️ comment says, spelling the names exactly as this line "
                    "uses them, and run the cell again. If the training was stopped, run it again and let it "
                    "finish, or open <b>Show answer</b> under the cell.")
        if name and hasattr(sys.modules[__name__], name) and not name.startswith("_"):
            tips.append(f"Did you mean <code>mlmath.{name}</code>?")
        close = difflib.get_close_matches(name, known, n=1, cutoff=0.75) if name else []
        if close and close[0] != name:
            tips.append(f"Did you mean <code>{html.escape(close[0])}</code>? Spelling and capitals must match exactly.")
        tips.append(f"Python does not know the name <code>{html.escape(name)}</code>. If an earlier cell makes it "
                    "(like <code>models</code>, <code>data</code>, <code>train</code> or <code>history</code>), "
                    "run the cells above this one first, in order. If you meant a number, type digits, like "
                    "<code>50</code>. If you meant text, put it in quotation marks.")
        return " ".join(tips)
    if issubclass(etype, AttributeError):
        if "'NoneType' object" in message:
            return ("The cell that should have made this did not finish, so it holds nothing. Go back to it, "
                    "read the message it printed, fix it, and run it again.")
        name, obj = getattr(value, "name", None), getattr(value, "obj", None)
        if name and obj is not None:
            close = difflib.get_close_matches(name, [a for a in dir(obj) if not a.startswith("_")], n=1, cutoff=0.6)
            if close:
                return f"Did you mean <code>{html.escape(close[0])}</code>? Spelling and capitals must match exactly. {_SHOW_ANSWER}"
        return f"Check the spelling of the name after the dot. {_SHOW_ANSWER}"
    if issubclass(etype, KeyError):
        models = namespace.get("models")
        names = ", ".join(f"<code>\"{html.escape(str(k))}\"</code>" for k in models) if isinstance(models, dict) else ""
        return ("A name in square brackets must match exactly, spaces and capitals included"
                + (f": {names}." if names else ".") + f" {_SHOW_ANSWER}")
    if issubclass(etype, TypeError):
        if "cannot be interpreted as an integer" in message:
            return "Use a whole number without a decimal point: <code>100</code>, not <code>100.0</code>."
        if "can only concatenate str" in message or "must be str, not" in message:
            return "Type numbers without quotation marks: <code>epochs=100</code>, not <code>epochs=\"100\"</code>."
        if "'str' object is not callable" in message:
            return "Leave out the quotation marks around the encoding: <code>mlmath.encode_digits</code>."
        if "encode_" in message and "missing" in message:
            return ("Leave out the brackets after the encoding: write <code>mlmath.encode_digits</code>, "
                    "not <code>mlmath.encode_digits()</code>.")
        if "unexpected keyword argument" in message:
            return f"Check the spelling of the name before the <code>=</code> sign. {_SHOW_ANSWER}"
        if "positional argument" in message:
            return ("Check the commas: Python reads <code>1,999</code> as two numbers, 1 and 999. "
                    f"Type <code>1999</code>. {_SHOW_ANSWER}")
        if "cannot unpack" in message:
            return f"Keep both names on the left of the <code>=</code>: <code>model, history = train(...)</code>. {_SHOW_ANSWER}"
    return (f"{_SHOW_ANSWER} If this cell worked before, choose Runtime, then Restart session, "
            "and run the cells again from the top.")


def _show_hint(shell, etype, value, tb, tb_offset=None):
    """IPython calls this for every error: show the usual error, then the hint."""
    shell.showtraceback((etype, value, tb), tb_offset=tb_offset)
    try:
        hint = _hint(etype, value, shell.user_ns)
    except Exception:  # a hint is a courtesy; never let it get in the way
        return None
    if hint:
        from IPython.display import display
        text = re.sub(r"<[^>]+>", "", hint)
        display({"text/html": ('<div style="border-left:4px solid #eda100;background:#fff8e6;color:#0b0b0b;'
                               'padding:8px 12px;margin:4px 0;font-family:sans-serif;font-size:14px;'
                               'line-height:1.45"><b>What this probably means.</b> ' + hint + "</div>"),
                 "text/plain": "What this probably means. " + html.unescape(text)}, raw=True)
    return None


def _install_hints():
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        return
    if shell is not None and hasattr(shell, "set_custom_exc"):
        shell.set_custom_exc((Exception, KeyboardInterrupt), _show_hint)


_install_hints()
