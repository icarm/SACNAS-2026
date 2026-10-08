# Mini Course: Machine Learning for Mathematics

A three-hour, hands-on introduction for mathematicians with no machine learning background. SACNAS Modern Math Workshop, 2026.

We teach a small neural network to compute **n mod 6** from the digits of n. It gets no rules, only examples. Its error falls in two separate steps, and we open the network up to find out what it learned at each one. The answer turns out to be the Chinese Remainder Theorem, one factor at a time.

## Get started

You need a laptop, a web browser, and a Google account. Nothing to install.

1. Click the first badge below. It opens the notebook in Google Colab.
2. Click into the first cell and press **Shift + Enter**. Keep going down the page.
3. When the last cell prints **You are ready**, you are set.

| Notebook | When | Open |
| --- | --- | --- |
| Setup check | First 15 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/00_setup_check.ipynb) |
| Session 1: train the network | Session 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/01_train.ipynb) |
| Session 2: open the network up | Session 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/02_interpret.ipynb) |
| Explore: break it yourself, with buttons | Break, and the end of session 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/03_explore.ipynb) |

## Schedule

| Time | Session 1: build it and watch it learn |
| --- | --- |
| 0:00 | Setup |
| 0:15 | The game by hand: n mod 6 without dividing |
| 0:30 | The data, and why we hold half of it back |
| 0:45 | Three ways to show a number to a network. Two of them fail. |
| 1:05 | Train, and watch the loss fall in two steps |
| 1:25 | Questions for the break |

30 minute break.

| Time | Session 2: open it up |
| --- | --- |
| 0:00 | Load three snapshots of the network |
| 0:10 | Probe 1: ask it questions |
| 0:30 | Probe 2: change one digit and see if the answer moves |
| 0:50 | Probe 3: read the connection strengths |
| 1:10 | Break it yourself with the explore panel |
| 1:25 | What we did, and what it does not prove |

## If something goes wrong

- **Colab will not load, or the wifi is weak.** Pair up with a neighbor whose notebook is running. One laptop per pair is plenty.
- **A cell shows an error.** In the Colab menu choose Runtime, then Restart session and run all.
- **Training is taking too long.** Press the stop button next to the cell. Session 2 does not need your run: it loads snapshots that ship with this repo.
- **The explore buttons do not appear.** Use the cells under "No buttons?" in the same notebook. They run the same experiments by letter.
- **You want to run on your own machine.** Download the repo (green Code button, Download ZIP), then:

  ```
  pip install -r requirements.txt
  jupyter lab
  ```

  and open the notebooks in the `notebooks` folder. A full training run takes about a minute on a laptop.

## What is in this repo

```
notebooks/      the four notebooks above
mlmath.py       helper code: data, plots, and the probes from session 2
lab.py          the button panel and experiments for the explore notebook
checkpoints/    the network at epochs 0, 30 and 400, plus its training history
solutions/      discussion notes with answers and measured results
tools/          scripts for the instructor
```

## For the instructor

Before the workshop:

1. Point the notebooks and this README at the real repo: `python tools/set_repo.py YOUR-NAME/YOUR-REPO`
2. Run the checks: `python tools/preflight.py` (about 3 minutes; add `--all` to also run every explore experiment). It fails if any notebook has saved outputs, so you cannot publish a spoiled or broken notebook by accident.
3. After pushing, open each Colab badge from a private browser window and run the first cell.

To regenerate the snapshots: `python tools/make_checkpoints.py`. To clear outputs after editing a notebook: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`.

`solutions/discussion_notes.md` has the measured results for every experiment and talking points for each question.

## License

MIT. See `LICENSE`.
