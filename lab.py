"""Experiments for notebooks/03_explore.ipynb.

    lab.panel()          buttons for every experiment
    lab.menu()           list the experiments and the settings you can change
    lab.run("D")         run one experiment without buttons
    lab.run("D", hide_digit=3)   the same experiment with one setting changed

Every experiment is the session 1 setup with a few settings changed.
"""
import difflib
from types import SimpleNamespace

import torch
import torch.nn as nn

import mlmath


def encode_base6(numbers, n_digits=4):
    """Write n in base 6 (six digits cover 0 to 9999) and one-hot encode each digit."""
    numbers = torch.as_tensor(numbers)
    digits = torch.stack([(numbers // 6 ** k) % 6 for k in reversed(range(6))], dim=1)
    return nn.functional.one_hot(digits, num_classes=6).float().reshape(len(numbers), -1)


ENCODINGS = {
    "one-hot digits": mlmath.encode_onehot,
    "digit values": mlmath.encode_digits,
    "raw number": mlmath.encode_raw,
    "one-hot base 6 digits": encode_base6,
}
PLACES = {"nowhere": None, "thousands": 0, "hundreds": 1, "tens": 2, "ones": 3}
LEARNING_RATE = {"SGD": 0.1, "Adam": 0.001}

SESSION_1 = dict(encoding="one-hot digits", modulus=6, width=512, optimizer="SGD", weight_decay=0.0,
                 train_frac=0.5, hide_place="nowhere", hide_digit=7, epochs=400, seed=0)

CHOICES = dict(
    encoding=list(ENCODINGS),
    modulus=list(range(2, 13)),
    width=[8, 16, 32, 64, 128, 256, 512, 1024],
    optimizer=["SGD", "Adam"],
    weight_decay=[0.0, 0.1, 1.0, 3.0],
    train_frac=[0.1, 0.25, 0.5, 0.8],
    hide_place=list(PLACES),
    hide_digit=list(range(10)),
    epochs=[50, 100, 200, 300, 400, 800],
    seed=list(range(10)),
)

PRESETS = {
    "A": ("Narrow network", dict(width=16),
          "Session 1 used 512 hidden units. With only 16, will it still find the digit-sum rule? "
          "Afterwards, try other widths."),
    "B": ("Adam optimizer", dict(optimizer="Adam", epochs=200),
          "Adam is the training rule most people use. It adapts the size of each nudge. "
          "What happens to the flat stretch? Does the network end up somewhere different, or only get there faster?"),
    "C": ("Less data", dict(train_frac=0.1),
          "Train on one tenth of the numbers instead of half. Which rule survives, and which turns into memorizing? "
          "Afterwards, try 25%."),
    "D": ("A digit it never saw", dict(hide_place="hundreds", hide_digit=7),
          "No training number has a 7 in the hundreds place. After training we test on exactly those numbers. "
          "Guessing would score 17%. What will the network score? Will it at least get parity?"),
    "E": ("Three rules: mod 12", dict(modulus=12),
          "n mod 12 needs parity (last digit), mod 4 (last two digits) and mod 3 (digit sum). "
          "How many steps will the loss take, and in what order will the rules arrive?"),
    "F": ("Other moduli", dict(modulus=9),
          "Mod 9 is the digit sum again, but now the whole sum mod 9 matters. Afterwards change n mod to 5, 11 or 7. "
          "Before each run, guess: fast, slow, or fail?"),
    "G": ("Rescue digit values", dict(encoding="digit values", optimizer="Adam", epochs=300),
          "Digit values went nowhere in session 1. With Adam and more time, does the network learn the rule "
          "or memorize? Compare the dashed curve (training numbers) with the solid ones (test numbers)."),
    "H": ("Base 6", dict(encoding="one-hot base 6 digits"),
          "Write every number in base 6 instead of base 10. Now n mod 6 is just the last digit. "
          "What happens to the two steps?"),
    "I": ("Heavy regularization", dict(optimizer="Adam", weight_decay=3.0, epochs=200),
          "Weight decay pulls every connection strength toward zero, which penalizes complicated solutions. "
          "At strength 3, can the network still afford the mod 3 rule? Afterwards, try strength 1."),
}

_panel = None
last = None  # the most recent experiment: last.model, last.data, last.history, last.settings


def settings_for(preset=None, **changes):
    """The session 1 settings, then the preset's changes, then yours (as typed, not yet checked)."""
    settings = dict(SESSION_1)
    if preset is not None:
        settings.update(PRESETS[_preset_letter(preset)][1])
    for key, value in changes.items():
        settings[_setting_name(key)] = value
    return settings


# Other names people reach for, and the setting they mean.
ALIASES = {"mod": "modulus", "m": "modulus", "n_mod": "modulus", "nmod": "modulus",
           "hidden_units": "width", "units": "width", "hidden": "width", "size": "width",
           "train": "train_frac", "frac": "train_frac", "fraction": "train_frac", "train_fraction": "train_frac",
           "epoch": "epochs", "wd": "weight_decay", "decay": "weight_decay",
           "place": "hide_place", "hide": "hide_place", "digit": "hide_digit", "hidden_digit": "hide_digit",
           "opt": "optimizer", "optimiser": "optimizer", "encode": "encoding", "encoder": "encoding"}
# Whole-number settings and the range allowed, so nobody starts an hour-long run by accident.
WHOLE = dict(modulus=(2, 100), width=(1, 4096), epochs=(1, 2000), hide_digit=(0, 9), seed=(0, 10 ** 9))


def _preset_letter(preset):
    if isinstance(preset, str):
        letter = "".join(c for c in preset if c.isalpha()).upper()
        if letter in PRESETS:
            return letter
    if isinstance(preset, (int, float)) and not isinstance(preset, bool):
        raise mlmath.Unreadable(f"Write the name of the setting too: lab.run(modulus={preset:g}), "
                                f"not lab.run({preset:g}).")
    raise mlmath.Unreadable(f"There is no experiment {preset!r}. The letters are {', '.join(PRESETS)}. "
                            "lab.menu() lists them.")


def _setting_name(key):
    name = key.strip().lower().replace(" ", "_").replace("-", "_")
    name = ALIASES.get(name, name)
    if name in SESSION_1:
        return name
    close = difflib.get_close_matches(name, list(SESSION_1), n=1, cutoff=0.5)
    raise mlmath.Unreadable(f"There is no setting called {key!r}."
                            + (f" Did you mean {close[0]}?" if close else "")
                            + f" The settings are: {', '.join(SESSION_1)}.")


def _clean(key, value):
    """Read one setting the way a person might type it, or explain what went wrong."""
    if value is Ellipsis:
        raise mlmath.Unreadable(f"Replace the ... after {key}= with a value and run again.")
    if key in WHOLE:
        low, high = WHOLE[key]
        n = mlmath._as_int(value)
        if not low <= n <= high:
            raise mlmath.Unreadable(f"{key} should be a whole number from {low} to {high}, not {n}.")
        return n
    if key == "train_frac":
        x = mlmath._as_number(value, "a fraction")
        if isinstance(value, str) and "%" in value or 1 < x <= 100:
            x = x / 100  # typed as a percent: 10 or "10%" means 0.1
        if not 0.01 <= x <= 0.95:
            raise mlmath.Unreadable("train_frac is the fraction of numbers to train on, from 0.01 to 0.95. "
                                    "For example 0.1 means 10%.")
        return x
    if key == "weight_decay":
        x = mlmath._as_number(value)
        if not 0 <= x <= 10:
            raise mlmath.Unreadable("weight_decay should be a number from 0 to 10.")
        return x
    if key == "optimizer":
        name = str(mlmath._unwrap(value)).strip().lower()
        for option in LEARNING_RATE:
            if name == option.lower():
                return option
        raise mlmath.Unreadable('optimizer should be "SGD" or "Adam", in quotation marks.')
    if key == "encoding":
        value = mlmath._unwrap(value)
        if callable(value):  # a function passed directly: register it under its own name
            ENCODINGS.setdefault(getattr(value, "__name__", "my encoding"), value)
            return next(name for name, f in ENCODINGS.items() if f is value)
        squash = lambda text: "".join(c for c in str(text).lower() if c.isalnum())  # "One-hot digits" -> "onehotdigits"
        name = squash(value)
        exact = [option for option in ENCODINGS if squash(option) == name]
        partial = [option for option in ENCODINGS if name and name in squash(option)]
        if exact or len(partial) == 1:
            return (exact or partial)[0]
        close = difflib.get_close_matches(str(value).lower(), list(ENCODINGS), n=1, cutoff=0.5)
        raise mlmath.Unreadable(f"There is no encoding {value!r}."
                                + (f' Did you mean "{close[0]}"?' if close else "")
                                + " The encodings are: " + ", ".join(f'"{e}"' for e in ENCODINGS) + ".")
    if key == "hide_place":
        value = mlmath._unwrap(value)
        if value is None or str(value).strip().lower() in ("nowhere", "none", "no", "", "false"):
            return "nowhere"
        return mlmath.position_name(mlmath._position(value))
    return value


def _title(settings):
    for letter, (name, _, _) in PRESETS.items():
        if settings == settings_for(letter):
            return f"{letter}. {name}: n mod {settings['modulus']}"
    if settings == SESSION_1:
        return "Session 1 settings: n mod 6"
    return f"Your experiment: n mod {settings['modulus']}"


@mlmath._forgiving
def run(preset=None, live=True, **changes):
    """Run one experiment. preset is a letter from menu(); changes override single settings.

    Returns the result (result.model, result.data, result.history, result.settings),
    or None after printing what to fix when a setting cannot be read.
    """
    global last
    settings = settings_for(preset, **changes)
    settings = {key: _clean(key, value) for key, value in settings.items()}
    place = PLACES[settings["hide_place"]]
    data = mlmath.make_data(modulus=settings["modulus"], train_frac=settings["train_frac"],
                            holdout=None if place is None else (place, settings["hide_digit"]))
    model, history, _ = mlmath.train(
        ENCODINGS[settings["encoding"]], data, width=settings["width"],
        optimizer=settings["optimizer"].lower(), lr=LEARNING_RATE[settings["optimizer"]],
        weight_decay=settings["weight_decay"], epochs=settings["epochs"], seed=settings["seed"],
        live="plot" if live else False, title=_title(settings), detail=True)
    mlmath.summarize(history, factors=True)
    if place is not None:
        held, truth = data.numbers[data.held], data.labels[data.held]
        answers = mlmath.predict_all(model, held)
        print(f"\nOn the {len(held)} hidden numbers, the ones with a {settings['hide_digit']} "
              f"in the {settings['hide_place']} place:")
        print(f"  exactly right: {(answers == truth).float().mean():.0%}")
        for d in mlmath.divisors(data.modulus):
            print(f"  right mod {d}:   {(answers % d == truth % d).float().mean():.0%}")
    last = SimpleNamespace(model=model, data=data, history=history, settings=settings)
    return last


def menu():
    """Print the experiments and the settings you can change."""
    for letter, (name, changes, question) in PRESETS.items():
        print(f"{letter}. {name}  {changes}")
        print(f"   {question}\n")
    print("Settings you can change, with the session 1 value first:")
    for key, options in CHOICES.items():
        print(f"  {key} = {SESSION_1[key]!r}   options: {options}")


# ---------------------------------------------------------------- buttons


def panel():
    """Show the experiment panel."""
    try:
        import ipywidgets as W
    except ImportError:
        print("The buttons need the ipywidgets package, which is missing here.")
        print('Use the cells below instead, for example lab.run("D").')
        return
    from IPython.display import clear_output, display

    style = {"description_width": "105px"}
    box = W.Layout(width="290px")
    labels = {"train_frac": [("10%", 0.1), ("25%", 0.25), ("50%", 0.5), ("80%", 0.8)]}
    names = dict(encoding="Encoding", modulus="n mod", width="Hidden units", optimizer="Optimizer",
                 weight_decay="Weight decay", train_frac="Train on", hide_place="Hide a digit in",
                 hide_digit="Digit to hide", epochs="Epochs", seed="Seed")
    controls = {key: W.Dropdown(options=labels.get(key, CHOICES[key]), value=SESSION_1[key],
                                description=names[key], style=style, layout=box)
                for key in CHOICES}

    question = W.HTML("<i>Pick an experiment, make a prediction, then press Run.</i>")
    run_button = W.Button(description="Run", button_style="success", icon="play",
                          layout=W.Layout(width="140px"))
    out = W.Output()
    # The training plot lives in a picture widget whose image is swapped in place, so it never flashes.
    picture = W.Image(format="png", layout=W.Layout(display="none", max_width="100%", height="auto"))

    def choose(letter):
        def handler(_):
            settings = settings_for(letter)
            for key, widget in controls.items():
                widget.value = settings[key]
            if letter is None:
                question.value = "<b>Session 1 settings.</b> The run from session 1, for comparison."
            else:
                name, _, text = PRESETS[letter]
                question.value = (f"<b>{letter}. {name}.</b> {text}<br>"
                                  "<i>Write down your prediction, then press Run.</i>")
        return handler

    preset_buttons = []
    for letter, (name, _, _) in PRESETS.items():
        button = W.Button(description=f"{letter}. {name}", layout=W.Layout(width="200px"))
        button.on_click(choose(letter))
        preset_buttons.append(button)
    reset = W.Button(description="Session 1 settings", layout=W.Layout(width="200px"))
    reset.on_click(choose(None))

    def on_run(_):
        run_button.disabled, run_button.description = True, "Training..."
        picture.layout.display = "block"
        mlmath._live["image"] = picture
        try:
            with out:
                clear_output(wait=True)  # clear only when new output arrives
                print("Training...")
                try:
                    run(**{key: widget.value for key, widget in controls.items()})
                except Exception as error:  # show the problem instead of failing silently
                    print("Something went wrong:", repr(error))
        finally:
            mlmath._live["image"] = None
            run_button.disabled, run_button.description = False, "Run"
    run_button.on_click(on_run)

    number = W.BoundedIntText(value=4712, min=0, max=9999, description="Number",
                              style={"description_width": "55px"}, layout=W.Layout(width="150px"))
    place = W.Dropdown(options=[(p, PLACES[p]) for p in ("thousands", "hundreds", "tens", "ones")],
                       value=0, description="Change the", style={"description_width": "75px"},
                       layout=W.Layout(width="200px"))
    ask_button = W.Button(description="Ask about it", layout=W.Layout(width="140px"))
    confusion_button = W.Button(description="Confusion table", layout=W.Layout(width="140px"))
    swap_button = W.Button(description="Swap table", layout=W.Layout(width="140px"))
    probe_out = W.Output()

    def probe(action):
        def handler(_):
            with probe_out:
                clear_output(wait=True)  # clear only when new output arrives
                if last is None:
                    print("Run an experiment first.")
                    return
                try:
                    action()
                except Exception as error:
                    print("Something went wrong:", repr(error))
        return handler

    this = lambda: {"this network": last.model}
    ask_button.on_click(probe(lambda: mlmath.ask(this(), number.value)))
    confusion_button.on_click(probe(lambda: mlmath.plot_confusions(this(), last.data)))
    swap_button.on_click(probe(lambda: mlmath.plot_swap_tables(this(), last.data, place.value)))

    order = ["encoding", "modulus", "width", "optimizer", "weight_decay",
             "train_frac", "hide_place", "hide_digit", "epochs", "seed"]
    settings_grid = W.GridBox([controls[key] for key in order],
                              layout=W.Layout(grid_template_columns="repeat(2, 300px)"))
    global _panel  # kept so tools/ can test the buttons
    _panel = W.VBox([
        W.HTML("<h3 style='margin:0'>1. Pick an experiment</h3>"),
        W.GridBox(preset_buttons + [reset], layout=W.Layout(grid_template_columns="repeat(3, 210px)")),
        question,
        W.HTML("<h3 style='margin:8px 0 0 0'>2. Check the settings (change any you like)</h3>"),
        settings_grid,
        W.HTML("<h3 style='margin:8px 0 0 0'>3. Predict, then run</h3>"),
        run_button,
        picture,
        out,
        W.HTML("<h3 style='margin:8px 0 0 0'>4. Probe the network you just trained</h3>"),
        W.HBox([number, ask_button]),
        W.HBox([confusion_button, place, swap_button]),
        probe_out,
    ])
    display(_panel)
