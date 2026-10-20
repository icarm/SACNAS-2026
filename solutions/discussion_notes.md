# Discussion notes

**Spoiler warning: this file gives away the ending of session 2.**

Answers and talking points for the questions in the notebooks, with results measured when the course was built (PyTorch 2.14 on CPU, seed 0). Other machines and PyTorch versions can differ in the last digit, and the more fragile runs (marked below) by more. The shapes do not change.

## The arc

The Chinese Remainder Theorem is the punchline, so nothing before the last section of session 2 names it. Session 1 never mentions parity or mod 3, and its training plot shows only loss and exact accuracy. Session 2 finds parity (Part 1), then the classes mod 3 (Part 2), then has students write the recipe that combines them (Part 3), and only then names the theorem and redraws the session 1 plot with the hidden mod 2 and mod 3 curves.

Students type in every ✏️ cell. Each one has a **Show answer** fold under it, and the answers also live in the cell metadata so `tools/preflight.py` can run the notebooks filled in. The helpers read typed values loosely (`2`, `2.0`, `"2"`, `2,` and `"50%"` all work; `"Thousands"` and `"the tens place"` are read as places) and print a plain message instead of crashing when they cannot, including for an unfilled `...`. Any error that does happen, syntax errors included, gets a yellow "What this probably means" box under it: importing `mlmath` registers the handler with IPython. It recognizes the common slips: a missing comma or bracket, `50%`, `0007`, `mod` instead of `%`, curly quotes, wrong indentation, a misspelled name (with a suggestion), a stopped training run, and a training line not typed yet. `lab.run` reads its settings the same way and refuses runs out of range, such as 100,000 epochs.

Session 2 does not depend on anything a student did in session 1. It loads the saved snapshots, and in Colab it runs on its own fresh runtime.

## Session 1 (01_train)

**Warm-up.** 1234 mod 6 = 4, 9000 mod 6 = 0, 5555 mod 6 = 5, 2026 mod 6 = 4. Most people use the last digit for parity and the digit sum for mod 3. Do not say so. The students write their method down, and session 2 hands it back at the end.

**The loop.** The two missing lines are `loss.backward()` and `optimizer.step()`. `mlmath.check_loop(train)` runs one epoch and confirms the loss falls (1.79 to 1.58). Anyone hopelessly stuck can run `train = mlmath.backup_train`.

**Experiment 1, raw number.** After 100 epochs: 17% on test numbers, which is guessing. Neighboring inputs 0.4711, 0.4712, 0.4713 have labels 1, 2, 3. A small network computes a function that changes slowly with its input, and this target changes with every step of 0.0001.

**Experiment 2, four digit values.** After 100 epochs: 17% on test numbers. Also stuck, even though everything is in the input. The encoding says 3 lies between 2 and 4, which is true for size and useless for remainders. (If someone asks: whether a digit is even flips ten times between 0 and 9.) Explore's button G gives this encoding Adam and 300 epochs: 21% on test numbers, 36% on training numbers. Training above test means it is memorizing.

**Experiment 3, one-hot.** Measured on the test numbers, as printed by `mlmath.look_at`:

| Epoch | Loss | e^loss | Exactly right |
| --- | --- | --- | --- |
| 0 | 1.79 | 6.0 | 19% |
| 1 | 1.59 | 4.9 | 33% |
| 2 | 1.35 | 3.9 | 31% |
| 10 | 1.14 | 3.1 | 31% |
| 30 | 1.17 | 3.2 | 32% |
| 60 | 1.14 | 3.1 | 37% |
| 100 | 0.57 | 1.8 | 80% |
| 200 | 0.13 | 1.1 | 97% |
| 400 | 0.05 | 1.1 | 99% |

**Reading the plot.**

- The loss starts at ln 6 = 1.79: belief spread evenly over 6 answers.
- The flat stretch sits at ln 3 = 1.10, and e^loss is about 3. The network has ruled out three of the six answers for every number and is undecided among the other three.
- The second drop is a slope, not a cliff. It takes about 60 epochs. The stretched axis makes it look steeper than it is, which is worth saying out loud.

**Questions for the break.** (1) It is not guessing at random: it scores one in three because it narrows every number to three answers. Let students guess which three; session 2 answers it. (2) and (3) are answered in session 2. For (3), good answers are "ask it about specific numbers", "check whether its wrong answers have something in common", and "change one digit and see whether the answer changes". Those are exactly the probes in session 2.

## Session 2 (02_interpret)

**Ask it.** At epoch 30 the network spreads its belief over three answers of the same parity as the truth and puts nothing on the other three: 4712 gets 23%, 46%, 30% on 0, 2, 4. It is often wrong (2025 and 3333 get answer 1), but never with the wrong parity.

**Right mod d.** On the 5,000 test numbers:

| d | Epoch 30 | Epoch 400 | Guessing |
| --- | --- | --- | --- |
| 2 | 100% | 100% | 50% |
| 3 | 32% | 99% | 33% |
| 6 | 32% | 99% | 17% |

If a student types a d that does not divide 6, the cell says so and suggests 2, 3 or 6.

**Change the thousands digit.** Fraction of answers unchanged when s is added to the thousands digit, over all 10,000 numbers:

| s | Epoch 30 | Epoch 400 | True answer |
| --- | --- | --- | --- |
| 1 | 90% | 1% | changes |
| 2 | 91% | 1% | changes |
| 3 | 89% | 99% | stays |
| 4 | 92% | 1% | changes |
| 5 | 92% | 0% | changes |
| 6 | 89% | 99% | stays |

At epoch 30 nothing you do to the thousands digit matters, because the network ignores it: it only knows parity, which lives in the last digit. At epoch 400 the answer survives exactly the shifts by 3 and 6, since 3000 and 6000 are divisible by 6.

**Groups.** `[[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]` scores 99% within groups and 1% across, in the thousands, hundreds and tens places alike. The early-finisher challenge, the ones digit, has groups `[[0, 6], [1, 7], [2, 8], [3, 9], [4], [5]]` (99% and 0%): the ones digit d decides parity and also adds d to the digit sum, so what matters is d mod 2 and d mod 3 together, which is d mod 6. That is the theorem in miniature, and a nice thing to point at after the reveal. A common wrong first guess is even and odd digits, `[[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]`, which scores 20% and 32%.

**The recipe.** `parity = last_digit % 2` and `r3 = digit_sum % 3`. It agrees with n mod 6 on all 10,000 numbers and with the epoch 400 network on 9,952 of them. Unfilled blanks give an IndexError at n = 0, which the cell catches and explains.

**The reveal.** Each of the six pairs (a mod 2, a mod 3) appears exactly once for a = 0, ..., 5. The Sunzi puzzle (threes leave 2, fives leave 3, sevens leave 2) has smallest answer 23. The closing plot is the session 1 run with the mod 2 and mod 3 curves added: parity hits 100% at epoch 1, mod 3 stays at guessing until about epoch 60 and then rises with the exact curve, which it equals from then on, since parity is already perfect.

Worth saying if there is time: the network learned the easy factor first because parity depends on one digit, while the remainder mod 3 depends on all four digits together and no single digit says anything about it.

## Explore (03_explore)

Results at the settings in the notebook:

| Experiment | Measured result |
| --- | --- |
| n mod 12 | 96.5% after 400 epochs. Three steps: parity by epoch 2, mod 4 by epoch 20 (loss down to about ln 3), a long flat stretch, then mod 3. Exactly the pieces 12 = 4 × 3 predicts. |
| n mod 10 | 100% after one epoch. Both pieces, mod 2 and mod 5, need only the last digit. |
| n mod 9 | 91.1% after 400 epochs, first above 90% at epoch 338. Mod 9 cannot be split, so there is one long climb. |
| n mod 7 | 95.9%, first above 90% at epoch 183. |
| n mod 15 | 97.0%. Mod 5 is perfect after one epoch (last digit), then a flat stretch at one in three, then mod 3. |
| n mod 5 | 100% after one epoch. |
| n mod 11 | 84.0% after 400 epochs. |
| Hidden 7 in the hundreds place | 99% on ordinary test numbers. On the 525 numbers with a 7 in the hundreds place: 3% exactly right, 100% parity, 3% mod 3. |
| `width=16` (fragile) | 68% on test numbers after 400 epochs. Parity 100%, mod 3 incomplete. An earlier PyTorch version gave 85.5%, so expect this one to vary. |
| `train_frac=0.1` | 100% on training numbers, 32% on test numbers. Parity still 100%: mod 3 is replaced by memorizing. With 25%, test accuracy is 84%. |
| `optimizer="Adam", epochs=200` | 100% on test numbers. The flat stretch shrinks: above 90% by epoch 27. |
| `encoding="one-hot base 6 digits"` | 100% after one epoch. No flat stretch at all. |
| `optimizer="Adam", weight_decay=3.0, epochs=200` | Parity 100% by epoch 1, then stuck: 59% at epoch 200. With weight decay 1 it reaches 97.5%. |

**The hidden digit is the one to dwell on.** The score on hidden numbers is 3%, far below the 17% from guessing. The network is not confused, it is confidently and consistently wrong: asked about 4712 it says 4 (wrong) with 53%, and about 4012 it says 4 (right) with 100%. The switch for "7 in the hundreds place" was never on during training, so its connections never moved from their small starting values. Measured: 68% of the time the network answers as if that 7 were a 0, 3, 6 or 9 (as if it added nothing to the digit sum), 28% as if it were a 2, 5 or 8, and only 3% correctly. It still gets parity, because parity never depended on that digit. A mathematician who knows the digit-sum rule applies it to any digit in any place. The network learned a separate table of digit classes for each place, and a table has no entry for something it never saw. The ordinary test score of 99% gave no warning.

**Other moduli.** Every one of these is a weighted digit sum: n mod m is the sum of each digit times 10^k mod m. For 3 and 9 every weight is 1. For 11 the weights alternate 1, -1. For 7 they are 1, 3, 2, 6 (ones digit first), because 10 is 3 mod 7. Moduli that factor into pieces depending on few digits (10, 5, 15's mod 5 piece) are easy. How "nice" the rule looks to a person does not predict how hard it is for the network: 7 is learned faster than 9, and 11 is slowest.

**Base 6.** In base 6 the answer is the last digit, so the problem is as easy as parity was. The encoding did all the work. A good follow-up: in base 6, how would n mod 3 or n mod 4 behave?

**Weight decay.** It charges the network for large connection strengths. Parity is cheap, the mod 3 rule is expensive, and at strength 3 the network cannot afford the expensive one. It settles for partial credit.

The button panel (`lab.panel()`, at the end of the notebook) still has all nine presets A to I for anyone who prefers clicking.
