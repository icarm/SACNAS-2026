# Mini Course: Machine Learning for Mathematics

A three-hour, hands-on introduction for mathematicians with no machine learning background. SACNAS Modern Math Workshop, 2026.

We teach a small neural network to compute **n mod 6** from the digits of n. Its "prediction error" falls in two separate places, and we open the network up to find out what it learned at each one. What it learned turns out to be a piece of mathematics you already know. The name comes at the very end.

## How to get started

You need a laptop, a web browser, and a Google account. Nothing needs to be installed on your computer.

1. Click the Session 1 badge below (Cmd-click on a Mac or Ctrl-click on Windows opens it in a new tab, so this page stays open). It opens the notebook in Google Colab.
2. Click into the first cell and press **Shift + Enter**. Then do the same for the second cell.
3. When the second cell prints **You are ready**, you are set. Wait there until the session starts.

| Notebook | When | Open |
| --- | --- | --- |
| Session 1: train the network | First 15 minutes (setup), then session 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/01_train.ipynb) |
| Session 2: open the network up | After the break | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/02_interpret.ipynb) |
| Explore: change anything | Only if there is time left after session 2 (it gives away the ending) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/03_explore.ipynb) |

Session 2 starts fresh from a saved copy of the network, so it does not matter how session 1 went.

In cells marked ✏️ you type something before you run: a prediction, a missing line of code, a hypothesis to test. The comments in the cell say what to type, and a **Show answer** link under the cell has the answer. Questions marked 🗣️ are for talking over with the person next to you.

## If something goes wrong

- **Colab will not load, or the wifi is weak.** Pair up with a neighbor whose notebook is running. One laptop per pair is plenty.
- **A ✏️ cell shows an error.** Read the yellow "What this probably means" box under the error. Usually a blank is not filled in yet, or there is a typo. Compare with **Show answer** under the cell, fix it, and run the cell again.
- **Everything is tangled.** In the Colab menu choose Runtime, then Restart session, and run the cells again from the top.
- **Colab says you have too many sessions open.** Close the notebooks you are done with: in each one choose Runtime, then Disconnect and delete runtime.
- **Training is taking too long.** Press the stop button next to the cell. The **Show answer** link under that cell says how to load the saved run instead. Session 2 does not need your run.
- **You want to run on your own machine.** Download the repo (green Code button, Download ZIP), then:

  ```
  pip install -r requirements.txt
  jupyter lab
  ```

  and open the notebooks in the `notebooks` folder. A full training run takes under a minute on a laptop.

## What is in this repo

```
notebooks/      the three notebooks above
mlmath.py       helper code: data, plots, network pictures, the checks behind the ✏️ cells, and the probes from session 2
lab.py          the ready-made experiments (other moduli, network sizes, hidden digits)
checkpoints/    the network at epochs 30 and 400, plus its training history
solutions/      discussion notes with answers and measured results (spoils the ending)
tools/          scripts for the instructor
```

## License

MIT. See `LICENSE`.
