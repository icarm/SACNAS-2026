# Mini Course: Machine Learning for Mathematics

A three-hour, hands-on introduction for mathematicians with no machine learning background. SACNAS Modern Math Workshop, 2026.

We teach a small neural network to compute **n mod 6** from the digits of n. Its "prediction error" falls in two separate places, and we open the network up to find out what it learned at each one. This investigation leads us to rediscovering the Chinese Remainder Theorem (specifically for 6), one factor at a time.

## How to get started

You need a laptop, a web browser, and a Google account. Nothing needs to be installed on your computer.

1. Click the first badge below. It opens the notebook in Google Colab.
2. Click into the first cell and press **Shift + Enter**. Keep going down the page.
3. When the last cell prints **You are ready**.

| Notebook | When | Open |
| --- | --- | --- |
| Setup check | First 15 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/00_setup_check.ipynb) |
| Session 1: train the network | Session 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/01_train.ipynb) |
| Session 2: open the network up | Session 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/02_interpret.ipynb) |
| Explore: break it yourself | Break, and the end of session 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icarm/SACNAS-2026/blob/main/notebooks/03_explore.ipynb) |

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

## License

MIT. See `LICENSE`.
