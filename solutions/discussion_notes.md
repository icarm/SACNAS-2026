# Discussion notes

Answers and talking points for the questions in the notebooks, with results measured when the course was built (PyTorch on CPU, seed 0). Other machines can differ in the last digit. The shapes do not change.

## Session 1 (01_train)

**By hand.** 1234 mod 6 = 4, 9000 mod 6 = 0, 5555 mod 6 = 5, 2026 mod 6 = 4.

**Experiment 1, raw number.** After 150 epochs: 16.5% on test numbers, parity 50%, mod 3 at 33%. Pure guessing. Neighboring inputs 0.4711, 0.4712, 0.4713 have labels 1, 2, 3. A small network computes a function that changes slowly with its input, and this target changes with every step of 0.0001.

**Experiment 2, four digit values.** After 150 epochs: 16.7% on test numbers, parity 54%. Also stuck, even though parity needs only the last digit. As a function of the digit's value, "is it even" flips ten times between 0 and 9, and the encoding says 3 lies between 2 and 4. Experiment G in the explore notebook gives this encoding a stronger optimizer and more time: parity reaches 84%, training accuracy 36%, test accuracy 21%. Training above test means it is memorizing.

**Experiment 3, one-hot.** Measured on the test numbers:

| Epoch | Loss | Exactly right | Parity right | Right mod 3 |
| --- | --- | --- | --- | --- |
| 0 | 1.79 | 19% | 59% | 33% |
| 2 | 1.35 | 31% | 100% | 31% |
| 10 | 1.14 | 31% | 100% | 31% |
| 30 | 1.17 | 32% | 100% | 32% |
| 60 | 1.14 | 37% | 100% | 37% |
| 100 | 0.57 | 81% | 100% | 81% |
| 200 | 0.13 | 97% | 100% | 97% |
| 400 | 0.05 | 99% | 100% | 99% |

**Reading the plot.**

- The loss starts at ln 6 = 1.79. A network that spreads its belief evenly over k answers, one of which is right, has loss ln k. At the start k = 6.
- The flat stretch sits at ln 3 = 1.10. The network has ruled out three of the six answers for every number and is undecided among the other three.
- On the flat stretch parity is at 100% and mod 3 is at the guessing level. So the three answers it ruled out are the ones with the wrong parity.
- The dashed training curve pulls ahead of the test curve on the flat stretch (around epoch 55: 55% on training numbers, 33% on test numbers). The network starts memorizing individual training numbers before it finds the digit-sum rule. Once it finds the rule, test accuracy catches up.
- The second drop is a slope, not a cliff. It takes about 60 epochs. The stretched axis makes it look steeper than it is, which is worth saying out loud.

**Why parity first?** Parity depends on one digit. The remainder mod 3 depends on all four digits together: no single digit says anything about it. Simple structure that sits in one place is found first.

**Questions for the break.** (1) It is not guessing at random. It scores one in three because it knows parity and nothing else, and you can tell from which wrong answers it gives. Questions (2) and (3) are answered in session 2.

## Session 2 (02_interpret)

**Probe 1.** For 4712 at epoch 30 the network puts its belief on 0, 2 and 4 and none on 1, 3, 5. In the epoch 30 confusion table every cell where the true answer and the network's answer differ in parity is empty: no parity mistakes in 5,000 test numbers.

**Probe 2, the swap test.** Fraction of answers unchanged when the thousands digit changes, over all 10,000 numbers:

| Snapshot | Add 3 (answer should stay) | Add 1 (answer should change) |
| --- | --- | --- |
| Epoch 0 | 67% | 67% |
| Epoch 30 | 89% | 90% |
| Epoch 400 | 99% | 1% |

A network that computes n mod 6 gives 100% and 0%. At epoch 30 the two are equal: the network mostly ignores the thousands digit, so nothing you do to it matters. At epoch 400 it has sorted the digits into {0, 3, 6, 9}, {1, 4, 7}, {2, 5, 8}.

**Ones digit.** At epoch 30 the table is a checkerboard: any even digit can replace any even digit. At epoch 400 the interchangeable pairs are 0 and 6, 1 and 7, 2 and 8, 3 and 9, with 4 and 5 alone. The ones digit d decides parity and also adds d to the digit sum, so what matters is d mod 2 and d mod 3 together, which is d mod 6.

**Probe 3.** The ones-digit weights show the even/odd checkerboard at epoch 30 (similarity about +0.3 within a parity class and -0.5 across). At epoch 400 the other digits show the three classes mod 3 at about +0.1 to +0.2 within a class against -0.2 across. Visible, but faint. The behavioral test is much sharper than the weights, and that gap is itself a lesson about reading networks.

**Your turn.** 48 of the 5,000 test numbers are wrong at epoch 400. There is no clean rule for which ones. Several come in near-pairs (2012 and 2018, 5752 and 5753 and 5758 and 5759), which suggests a few digit combinations that are thin in the training half.

## Explore (03_explore)

Each button sets the session 1 settings plus the changes listed. Results at the preset settings:

| Button | Change | Measured result |
| --- | --- | --- |
| A. Narrow network | width 16 | 85.5% on test numbers after 400 epochs. Parity fine, mod 3 incomplete. |
| B. Adam optimizer | Adam, 200 epochs | 100% on test numbers. The flat stretch shrinks to about 10 epochs (90% by epoch 27). |
| C. Less data | train on 10% | 100% on training numbers, 32% on test numbers. Parity still 100%. Mod 3 is replaced by memorizing. With 25%, test accuracy is 84%. |
| D. A digit it never saw | no 7 in the hundreds place during training | 99% on ordinary test numbers. On the 525 numbers with a 7 in the hundreds place: 3% exactly right, 100% parity, 3% mod 3. |
| E. Three rules: mod 12 | modulus 12 | 96.5% after 400 epochs. Three steps: parity by epoch 2, mod 4 by epoch 20 (loss down to ln 3), a long flat stretch, then mod 3. |
| F. Other moduli | modulus 9 | 90.8% after 400 epochs. |
| G. Rescue digit values | digit values, Adam, 300 epochs | 21% on test numbers, 36% on training numbers, parity 84%. |
| H. Base 6 | one-hot base 6 digits | 100% after one epoch. No flat stretch at all. |
| I. Heavy regularization | Adam, weight decay 3, 200 epochs | Parity 100% by epoch 1, then stuck: 56% at epoch 200, wobbling between 50% and 67%. With weight decay 1 it reaches 97%. |

Other moduli, same settings as session 1, 400 epochs: mod 5 reaches 100% in one epoch, mod 7 reaches 95.9%, mod 9 90.8%, mod 11 84.1%.

**D is the one to dwell on.** The score on hidden numbers is 3%, far below the 17% you would get by guessing. The network is not confused, it is confidently and consistently wrong. The switch for "7 in the hundreds place" was never on during training, so its connections never moved from their small starting values. Measured: 68% of the time the network answers as if that 7 were a 0, 3, 6 or 9 (as if it added nothing to the digit sum), 29% of the time as if it were a 2, 5 or 8, and only 3% of the time correctly. It still gets parity, because parity never depended on that digit. A mathematician who knows the digit-sum rule applies it to any digit in any place. The network learned a separate table of digit classes for each place, and a table has no entry for something it never saw. The ordinary test score of 99% gave no warning. With the probes: Ask about 4712 and then 4012, and use the swap table on the hundreds place.

**F, other moduli.** Every one of these is a weighted digit sum: n mod m is the sum of each digit times 10^k mod m. For 3 and 9 every weight is 1. For 11 the weights alternate 1, -1. For 7 they are 1, 3, 2, 6 (ones digit first), because 10 is 3 mod 7. So the network faces the same kind of problem each time and difficulty grows with the number of residue classes it has to track: 7 is learned faster than 9, and 11 is slowest. How "nice" the rule looks to a person does not predict how hard it is for the network. Mod 5 needs only the last digit and is immediate.

**H, base 6.** In base 6 the answer is the last digit, so the problem is as easy as parity was. The encoding did all the work. Good follow-up: in base 6, how would n mod 3 or n mod 4 behave?

**I, regularization.** Weight decay charges the network for large connection strengths. Parity is cheap, the mod 3 rule is expensive, and at strength 3 the network cannot afford the expensive one. It settles for partial credit.
