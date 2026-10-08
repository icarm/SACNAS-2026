"""Check that the live demo will run on this machine. From the repo root:

    python tools/preflight.py          about 3 minutes
    python tools/preflight.py --all    also runs every explore experiment, about 8 more minutes

It checks the packages, checks that the saved snapshots tell the story the
notebooks describe, retrains the main network here and checks for the two
steps, and runs notebooks 00, 01 and 02 top to bottom without saving them.
"""
import math
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
failures = []


def check(ok, message):
    print(("  PASS  " if ok else "  FAIL  ") + message)
    if not ok:
        failures.append(message)


def note(message):
    print("  NOTE  " + message)


import platform  # noqa: E402
import subprocess  # noqa: E402


def under_rosetta():
    """True when this is an Intel build of Python running on an Apple Silicon Mac."""
    if sys.platform != "darwin":
        return False
    try:
        flag = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True)
        return flag.stdout.strip() == "1"
    except Exception:
        return False


print("0. This Python")
print(f"  NOTE  {sys.executable} ({platform.machine()}, Python {platform.python_version()})")
ROSETTA = under_rosetta()
if ROSETTA:
    print("  NOTE  This is an Intel build of Python running on an Apple Silicon Mac (through Rosetta).\n"
          "        It works, but training is slower, and any Python it launches also runs in Intel mode.\n"
          "        Prefer your Mac's native Python, for example: /usr/local/bin/python3 tools/preflight.py")

print("\n1. Packages")
for package in ("torch", "numpy", "matplotlib"):
    try:
        module = __import__(package)
        check(True, f"{package} {module.__version__}")
    except ImportError:
        check(False, f"{package} is missing. Run: pip install -r requirements.txt")
if failures:
    sys.exit("\nInstall the missing packages and run this again.")
try:
    import ipywidgets
    check(True, f"ipywidgets {ipywidgets.__version__} (only the explore notebook needs it)")
except ImportError:
    note("ipywidgets is missing, so the explore buttons will not show here. lab.run() still works.")

import mlmath  # noqa: E402
import lab  # noqa: E402

import json  # noqa: E402
for name in sorted(os.listdir("notebooks")):
    if name.endswith(".ipynb"):
        cells = json.load(open(f"notebooks/{name}"))["cells"]
        saved = sum(1 for c in cells if c.get("outputs"))
        check(saved == 0, f"{name} has no saved outputs" if saved == 0 else
              f"{name} has saved outputs in {saved} cells. Clear them before publishing: "
              "jupyter nbconvert --clear-output --inplace notebooks/*.ipynb")
bad = [(letter, key) for letter in lab.PRESETS for key, value in lab.settings_for(letter).items()
       if value not in lab.CHOICES[key]]
check(not bad, "every explore button uses settings the panel offers" + (f" (problems: {bad})" if bad else ""))

readme = open("README.md").read()
if "OWNER/" in readme:
    note("The Colab links still say OWNER. Run: python tools/set_repo.py YOUR-NAME/YOUR-REPO")
else:
    check(True, "Colab links point at a real repo name")

print("\n2. Saved snapshots (used by session 2)")
data = mlmath.make_data()
snap = {e: mlmath.load_model(f"checkpoints/mod6_epoch{e:03d}.pt") for e in (0, 30, 400)}
X, y = mlmath.encode_onehot(data.numbers), data.labels
row30 = mlmath.evaluate(snap[30], X, y, data)
row400 = mlmath.evaluate(snap[400], X, y, data)
check(row30["mod2_acc"] == 1.0, f"epoch 30 gets parity right on every test number ({row30['mod2_acc']:.1%})")
check(row30["test_acc"] < 0.4, f"epoch 30 is still near one in three ({row30['test_acc']:.1%})")
check(row400["test_acc"] > 0.98, f"epoch 400 is above 98% ({row400['test_acc']:.1%})")
stay30, move30 = mlmath.swap_test(snap[30], data, 0, 3), mlmath.swap_test(snap[30], data, 0, 1)
stay400, move400 = mlmath.swap_test(snap[400], data, 0, 3), mlmath.swap_test(snap[400], data, 0, 1)
check(abs(stay30 - move30) < 0.05, f"epoch 30 ignores the thousands digit (+3: {stay30:.0%}, +1: {move30:.0%})")
check(stay400 > 0.95 and move400 < 0.05, f"epoch 400 sorts digits mod 3 (+3: {stay400:.0%}, +1: {move400:.0%})")
history = mlmath.load_history("checkpoints/mod6_history.csv")
check(len(history) == 401, "training history has all 401 epochs")

print("\n3. Retrain the main network on this machine (same settings as session 1)")
start = time.time()
_, fresh, _ = mlmath.train(mlmath.encode_onehot, data, live=False)
seconds = time.time() - start
check(fresh[5]["mod2_acc"] == 1.0, f"parity learned by epoch 5 ({fresh[5]['mod2_acc']:.1%})")
check(abs(fresh[30]["train_loss"] - math.log(3)) < 0.15,
      f"epoch 30 loss sits near ln 3 = 1.10 ({fresh[30]['train_loss']:.2f})")
check(fresh[30]["test_acc"] < 0.4, f"epoch 30 test accuracy is still low ({fresh[30]['test_acc']:.1%})")
check(fresh[-1]["test_acc"] > 0.98, f"epoch 400 test accuracy above 98% ({fresh[-1]['test_acc']:.1%})")
note(f"training took {seconds:.0f} seconds here without the live plot; budget about 1.5 times that live")
drift = max(abs(a["test_acc"] - b["test_acc"]) for a, b in zip(fresh, history))
note(f"largest gap from the saved history at any epoch: {drift:.1%} (0% means this machine matches exactly)")

print("\n4. Run the notebooks top to bottom with this Python (outputs are not saved)")
try:
    import ipykernel  # noqa: F401
    import nbformat
    from nbclient import NotebookClient
except ImportError as missing:
    note(f"{missing.name} is missing, so the notebooks were not run. "
         f"Run: {sys.executable} -m pip install nbclient ipykernel")
else:
    # A throwaway kernel that runs this exact Python, whatever Jupyter's default kernel is.
    import json
    import tempfile
    kernel_home = tempfile.mkdtemp()
    os.makedirs(os.path.join(kernel_home, "kernels", "ml4math-preflight"))
    with open(os.path.join(kernel_home, "kernels", "ml4math-preflight", "kernel.json"), "w") as f:
        json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                   "display_name": "ml4math preflight", "language": "python"}, f)
    os.environ["JUPYTER_PATH"] = kernel_home + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    for name in ("00_setup_check", "01_train", "02_interpret"):
        notebook = nbformat.read(f"notebooks/{name}.ipynb", as_version=4)
        start = time.time()
        try:
            NotebookClient(notebook, timeout=900, kernel_name="ml4math-preflight",
                           resources={"metadata": {"path": "notebooks"}}).execute()
            check(True, f"{name} ran without errors in {time.time() - start:.0f} seconds")
        except Exception as error:
            check(False, f"{name} stopped with an error: {str(error).strip().splitlines()[-1]}")

print("\n5. Jupyter's default Python 3 kernel (what jupyter lab uses if you run the demo locally)")
try:
    import shutil
    import subprocess
    from jupyter_client.kernelspec import KernelSpecManager
    spec = KernelSpecManager().get_kernel_spec("python3")
    kernel_python = shutil.which(spec.argv[0]) or spec.argv[0]
except Exception as error:
    note(f"could not find Jupyter's default kernel ({error!r}). Skipping this check.")
else:
    probe = subprocess.run([kernel_python, "-c", "import torch, matplotlib, ipywidgets"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        check(True, f"the default kernel ({kernel_python}) has torch, matplotlib and ipywidgets")
    elif "incompatible architecture" in probe.stderr:
        check(False, f"the default kernel runs {kernel_python}, and its packages are built for a different chip\n"
                     f"        than the mode it was started in (Intel versus Apple Silicon). That happens when an\n"
                     f"        Intel-build Python, like an Intel Anaconda, starts it. Run everything with the native\n"
                     f"        Python instead:\n"
                     f"            {kernel_python} tools/preflight.py\n"
                     f"            {kernel_python} -m jupyter lab        (for the local demo)")
    else:
        missing = (probe.stderr.strip().splitlines() or ["unknown error"])[-1][:300]
        check(False, f"the default kernel runs {kernel_python}, which cannot import everything ({missing}).\n"
                     f"        The notebooks will fail if you run them locally with jupyter lab. Fix with:\n"
                     f"        {kernel_python} -m pip install -r requirements.txt")

if "--all" in sys.argv:
    print("\n6. Every explore experiment")
    for letter, (name, _, _) in lab.PRESETS.items():
        start = time.time()
        try:
            result = lab.run(letter, live=False)
            check(True, f"{letter}. {name}: {result.history[-1]['test_acc']:.1%} on test numbers "
                        f"({time.time() - start:.0f} s)")
        except Exception as error:
            check(False, f"{letter}. {name} failed: {error!r}")

print()
if failures:
    sys.exit(f"{len(failures)} check(s) failed. See FAIL lines above.")
print("All checks passed. The demo is ready on this machine.")
