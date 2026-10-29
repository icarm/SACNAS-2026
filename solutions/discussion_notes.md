# Discussion notes

**Spoiler warning: this file gives away the ending of session 2.**

Answers and talking points for the questions in the notebooks, with results measured when the course was built (PyTorch 2.14 on CPU, seed 0). Other machines and PyTorch versions can differ in the last digit, and the more fragile runs (marked below) by more. The shapes do not change.

## The arc

The Chinese Remainder Theorem is the punchline, so nothing before the last section of session 2 names it. Session 1 never mentions parity or mod 3, and its training plots show only loss and exact accuracy (plus training curves). Instead it lets students notice that some remainders are learned in one epoch and others take hundreds, and that a small network gets stuck on the flat stretch. Session 2 finds parity (Part 1), then the classes mod 3 (Part 2), then has students write the recipe that combines them (Part 3), and only then names the theorem and redraws the session 1 plot with the hidden mod 2 and mod 3 curves. Part 4 uses the theorem to predict new runs, Part 5 shows a 99% network that still misses the rule, and Part 6 asks students what they would train on a math question of their own.

Students type in every ✏️ cell. Each one has a **Show answer** fold under it, and the answers also live in the cell metadata so `tools/preflight.py` can run the notebooks filled in. 🗣️ marks a question for students to talk over with their neighbor; these are good moments to pause the room and take a few answers. The helpers read typed values loosely (`2`, `2.0`, `"2"`, `2,` and `"50%"` all work; `"Thousands"` and `"the tens place"` are read as places) and print a plain message instead of crashing when they cannot, including for an unfilled `...`. Any error that does happen, syntax errors included, gets a yellow "What this probably means" box under it: importing `mlmath` registers the handler with IPython. It recognizes the common slips: a missing comma or bracket, `50%`, `0007`, `mod` instead of `%`, curly quotes, wrong indentation, a misspelled name (with a suggestion), a letter typed without quotes, a stopped training run, and a training line not typed yet. `lab.run` and `lab.experiment` read their settings the same way and refuse runs out of range, such as 100,000 epochs.

Session 2 does not depend on anything a student did in session 1. It loads the saved snapshots, and in Colab it runs on its own fresh runtime.

## Pacing

Rough times for a room working at a normal pace. Each session is planned to fill its 90 minutes, so the Explore notebook is only a backup.

| Session 1 | Minutes | Session 2 | Minutes |
| --- | --- | --- | --- |
| 0–1. Setup, warm-up | 10 | Setup, break recap | 5 |
| 2–3. Data, encodings | 8 | Part 1: ask, mistakes, right mod d | 17 |
| 4. The loss | 12 | Part 2: thousands digit, groups, other digits | 22 |
| 5. What a network is, the training loop | 16 | Part 3 and the reveal | 15 |
| 6. Loss curves | 6 | Part 4: predictions | 16 |
| 7–8. Experiments 1 and 2 | 12 | Part 5: the hidden digit | 8 |
| 9–10. Experiment 3, read the plot | 13 | Part 6: what would you train? | 15 |
| 11. Other remainders | 10 | | |
| 12. A different network | 8 | | |

Both columns add up to a little over 90 minutes on purpose. If session 1 runs behind, cut Experiment 5. In session 2, protect Part 6, the closing discussion: if you're behind, skip "Your own" in Part 4 and then Part 5. Each training run takes about 15 seconds on a laptop and up to a minute on Colab; width 2048 takes about four times as long.

## Session 1 (01_train)

**Warm-up.** 1234 mod 6 = 4, 9000 mod 6 = 0, 5555 mod 6 = 5, 2026 mod 6 = 4. Most people use the last digit for parity and the digit sum for mod 3. Do not say so. The students write their method down, and session 2 hands it back at the end.

**The loss (section 4).** The three typed beliefs give ln 6 = 1.79 (even spread), ln 3 = 1.10 (a third each on 1, 2 and 5) and ln 2 = 0.69 (50/50 between 2 and any other answer). The 99%-sure-and-wrong network scores 4.61, much worse than knowing nothing. A belief of 0 on the right answer gives an infinite loss, which the cell says. Beliefs that do not add up to 100 are rescaled, with a note. The point to land: e^loss is the number of answers still in play, which is how the flat stretch gets read later.

**Loss curves (section 6).** The four made-up runs are A = learns suddenly, B = never learns, C = memorizes, D = learns steadily. Curve C is the one worth a minute: training loss to 0, test loss up past ln 6. Students meet the real thing in Experiment 4 (n mod 9).

**The tiny network (section 5).** Its strengths are set by hand, not trained. The five hidden units switch on at x = 0.3, 0.4, 0.6 and 0.7 (units 1 to 4) and off at x = 0.2 (unit 5, which only fires for small x), so those are the five places the curve can bend. At x = 0.2 every unit is off, at 0.5 units 1 and 2 are on (values 2.0 and 1.0, belief 99%), and at 0.9 units 1 to 4 are on (6, 5, 3, 2) and cancel out to a belief of 1%. The point to land: a hidden unit is a hinge, and the network's answer is built by adding hinges. That comes back right after Experiment 1.

**Our network.** Before training, `mlmath.draw_network(4712)` shows beliefs between 15% and 18% on all six answers, so the loss is about ln 6 = 1.79. The picture draws 16 of the 512 hidden units. The `nn.Sequential` line in the training cell is the same picture written in code.

**The loop.** The two missing lines are `loss.backward()` and `optimizer.step()`. `mlmath.check_loop(train)` runs one epoch and confirms the loss falls (1.79 to 1.58). Anyone hopelessly stuck can run `train = mlmath.backup_train`.

**Experiment 1, raw number.** After 100 epochs: 17% on test numbers, which is guessing. Neighboring inputs 0.4711, 0.4712, 0.4713 have labels 1, 2, 3. A small network computes a function that changes slowly with its input, and this target changes with every step of 0.0001.

The `compare_answers` picture under Experiment 1 makes this visible: for 4700 to 4729 the raw-value network answers 4 every time. With one input, each of the six scores is a hinged line with at most 512 bends, while following n mod 6 would need a peak at every sixth number, over 3,000 bends. Under Experiment 2, the digit-values network answers in blocks (2, 2, 2, 2, 2, 1, 1, 1, 1, 1, ...): it changes its answer only when the last digit moves a long way, which is the "nearby inputs, nearby outputs" habit in action.

**If someone asks why we divide.** Measured with the session 1 settings: fed 4712 unscaled, the network starts at a loss of 855 instead of 1.79, and after one epoch it is stuck at exactly ln 6 for good. It scores like guessing, but because the huge inputs knocked it out, not because of the problem. Unscaled digits (0 to 9) do no harm: 18.5% on test numbers after 100 epochs, still guessing.

**Experiment 2, four digit values.** After 100 epochs: 17% on test numbers. Also stuck, even though everything is in the input. The encoding says 3 lies between 2 and 4, which is true for size and useless for remainders. (If someone asks: whether a digit is even flips ten times between 0 and 9.) The explore notebook's digit-values suggestion (button G) gives this encoding Adam and 300 epochs: 22% on test numbers, 36% on training numbers. Training above test means it is memorizing.

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

**Experiment 4, other remainders.** Measured with `lab.experiment` (400 epochs, one-hot):

| Modulus | Test accuracy | First at 90% | Shape |
| --- | --- | --- | --- |
| 10 | 100% | epoch 1 | one cliff |
| 5 | 100% | epoch 1 | one cliff |
| 4 | 100% | epoch 8 | fast |
| 8 | 100% | epoch 62 | steps |
| 7 | 95.9% | epoch 183 | long flat stretch, then a climb |
| 9 | 91.1% | epoch 338 | test loss rises above ln 9 while training numbers are memorized, then falls |
| 11 | 84.0% | not in 400 | like 9, slower |
| 2 | 100% | epoch 1 | |
| 3 | 99.7% | epoch 91 | |

What to draw out, without naming parity or mod 3: the easy ones need only the last digit or two; the hard ones need every digit. n mod 6's first drop is as fast as n mod 10, and its second is slow like n mod 9. Someone will try 2 and 3. Let them; it is exactly the right experiment, and session 2 confirms it.

**Experiment 5, a different network.** n mod 6, 400 epochs:

| Network | Test accuracy | First at 90% |
| --- | --- | --- |
| 2 hidden units | 36% | never |
| 4 | 64% | never |
| 16 | 68% | never |
| 64 | 96.4% | epoch 216 |
| 512 (session 1) | 99.0% | epoch 122 |
| 2048 | 99.7% | epoch 96 |
| two layers of 16 | 100% | epoch 120 |
| two layers of 512 | 99.8% | epoch 77 |

Small networks all make the first drop and stall on or just past the flat stretch: 36% is the flat-stretch score. The small widths are fragile (8 units gets 44%, less than 4 units), so expect neighbors' numbers to differ if anything changes. Two layers of 16 beating one layer of 16 is the surprise; one good answer is that the second layer can combine what the first layer computed, so it does not have to do everything in one step.

**Questions for the break.** (1) It is not guessing at random: it scores one in three because it narrows every number to three answers. Let students guess which three; session 2 answers it. (2) and (3) are answered in session 2; the scoreboard hint in (2) points at "the fast part is a last-digit thing, the slow part is an every-digit thing". For (3), good answers are "ask it about specific numbers", "check whether its wrong answers have something in common", and "change one digit and see whether the answer changes". Those are exactly the probes in session 2.

## Session 2 (02_interpret)

**The pictures in Part 1.** `mlmath.draw_network(4712, models)` shows the two snapshots side by side: epoch 30 puts 23%, 46% and 30% on 0, 2 and 4, and epoch 400 puts 98% on 2. `compare_answers` for 4700 to 4729 shows epoch 30 answering 2, 1, 2, 1, ... down the whole row, right 10 times out of 30. The alternation is the parity, and it is the strongest hint before "right mod d".

**Ask it.** At epoch 30 the network spreads its belief over three answers of the same parity as the truth and puts nothing on the other three: 4712 gets 23%, 46%, 30% on 0, 2, 4. It is often wrong (2025 and 3333 get answer 1), but never with the wrong parity.

**Mistakes.** `mlmath.mistakes` lists 10 wrong test numbers at epoch 30 (it gets 3,405 of 5,000 wrong). Every wrong answer differs from the truth by 2 or 4, so it always has the right parity. That is the pattern to spot.

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

**Groups.** `[[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]` scores 99% within groups and 1% across, in the thousands, hundreds and tens places alike; the "other digits" cell checks the hundreds and tens. A common wrong first guess is even and odd digits, `[[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]`, which scores 20% and 32%.

**The ones digit.** Groups `[[0, 6], [1, 7], [2, 8], [3, 9], [4], [5]]` (99% and 0%). The ones digit d decides parity and also adds d to the digit sum, so what matters is d mod 2 and d mod 3 together, which is d mod 6. 2 and 8 share both; 2 and 5 share the remainder mod 3 but not parity. That is the theorem in miniature, one section before it is named.

**The recipe.** `parity = last_digit % 2` and `r3 = digit_sum % 3`. It agrees with n mod 6 on all 10,000 numbers and with the epoch 400 network on 9,952 of them. Unfilled blanks give an IndexError at n = 0, which the cell catches and explains.

**4 = 2 × 2.** The line prints only two distinct pairs, (0, 0) and (1, 1), for four values of a, so n mod 2 twice cannot tell 0 from 2. That is why the theorem needs no common factor, and it sets up 9 = 3 × 3 in Part 4.

**The recipe picture.** `mlmath.draw_recipe(4712)` works the recipe through for one number: last digit 2 gives n mod 2 = 0, digit sum 14 gives n mod 3 = 2, and the only answer from 0 to 5 with both is 2. Students can type their own numbers. `draw_pairs(6, 2, 3)` then shows the six boxes with one number each, and `draw_pairs(4, 2, 2)` shows two crowded boxes (0, 2 and 1, 3) and two empty ones. `draw_pairs(12, 4, 3)` in Part 4 is the same picture for 12.

**The reveal.** Each of the six pairs (a mod 2, a mod 3) appears exactly once for a = 0, ..., 5. The Sunzi puzzle (threes leave 2, fives leave 3, sevens leave 2) has smallest answer 23. The closing plot is the session 1 run with the mod 2 and mod 3 curves added: parity hits 100% at epoch 1, mod 3 stays at guessing until about epoch 60 and then rises with the exact curve, which it equals from then on, since parity is already perfect.

Worth saying if there is time: the network learned the easy factor first because parity depends on one digit, while the remainder mod 3 depends on all four digits together and no single digit says anything about it.

**Part 4, predictions.** Measured with `lab.run`, epoch at which each piece first reaches 90% right:

| Modulus | Pieces | Measured | Test accuracy |
| --- | --- | --- | --- |
| 10 = 2 × 5 | 2, 5 | both at epoch 1 | 100% |
| 12 = 4 × 3 | 2, 4, 3 | mod 2 at 1, mod 4 at 15, mod 3 at 186 | 96.5% |
| 30 = 2 × 3 × 5 | 2, 3, 5 | mod 2 and mod 5 at 1, mod 3 at 186 | 97.9% |
| 15 = 3 × 5 | 3, 5 | mod 5 at 1, mod 3 at 172 | 97.0% |
| 14 = 2 × 7 | 2, 7 | mod 2 at 1, mod 7 at 332 | 91.2% |
| 20 = 4 × 5 | 4, 5 | all by epoch 2 | 100% |
| 9 | 9 | mod 3 at 210, mod 9 at 338 | 91.1% |

n mod 30 is the showpiece: 30 answers, but once the last-digit pieces (2 and 5) are in, only 30 / 10 = 3 remain, so the flat stretch sits at ln 3 = 1.10, the same level as n mod 6. `look_at` prints e^loss of about 3.3 at epoch 30. n mod 12 shows three levels: ln 12, then ln 6 briefly after parity, then ln 3 after mod 4, then down. For 9, the theorem does not split it, yet a "right mod 3" curve still appears and comes first, because knowing n mod 9 includes knowing n mod 3. It is a step on the way, not a separate piece; worth a sentence if someone asks. The answer to "what decides how hard a piece is": how many digits it depends on. Pieces settled by the last digit or two (2, 4, 5, 8, 10, 20) are fast; 3, 7, 9 and 11 need every digit.

**Part 5, the hidden digit.** See the next paragraph under Explore for the full story; the measured numbers are the same. The `draw_network` picture at the end of Part 5 marks the hidden switch with a dashed orange outline. For 4712 that switch is on, and the network puts most of its belief on 4, the wrong answer.

**Part 6, what would you train?** This is the closing discussion, and the aim is to get students talking to each other and then to you. Give them about 5 minutes to fill in the box alone, 5 with a neighbor, then take a few pairs' questions in front of the room. Useful prompts while you circulate:

- *Where do the examples come from?* If you can compute the answer, you can make as many as you want. If you can't, the network's training data is the bottleneck.
- *Which encoding?* Session 1 is the warning: the same information in raw value, digit values and one-hot gave 17%, 17% and 99%. Ask what "one-hot" would mean for their inputs, such as polynomial coefficients or a graph's edges.
- *What would you look for inside?* Every probe from today carries over: ask it about examples, read its mistakes, score it "right mod d" (or right on some simpler property), change one part of the input, hide part of the input.
- *Prime or not?* A good one to discuss if it comes up. A network trained on n's digits can quickly rule out numbers ending in 0, 2, 4, 5, 6 or 8, since that only needs the last digit. Past that there's no digit shortcut for being prime, so expect it to stall (we haven't measured this one). A sum of two squares is a nice contrast: it is governed by primes that are 3 mod 4, which is structure a network might partly find.

The real-world examples in the notebook text: Davies et al., "Advancing mathematics by guiding human intuition with AI" (Nature, 2021), found a new relationship between algebraic and geometric invariants of knots, and Wagner, "Constructions in combinatorics via neural networks" (2021), found counterexamples to several conjectures in graph theory.

## Explore (03_explore)

The explore notebook is a backup for a room that finishes early. Its suggestions, with results at the settings in the notebook:

| Experiment | Measured result |
| --- | --- |
| `train_frac=0.1` | 100% on training numbers, 32% on test numbers. Parity still 100%: mod 3 is replaced by memorizing. With 25%, test accuracy is 84%. |
| `optimizer="Adam", epochs=200` | 100% on test numbers. The flat stretch shrinks: above 90% by epoch 27. |
| `encoding="one-hot base 6 digits"` | 100% after one epoch. No flat stretch at all. |
| `optimizer="Adam", weight_decay=3.0, epochs=200` | Parity 100% by epoch 1, then stuck: 59% at epoch 200. With weight decay 1 it reaches 97.5%. |
| `hide_place="ones", hide_digit=7` | 99% on ordinary test numbers. On the 485 hidden numbers: 7% exactly right, and parity only 52%, a coin flip, because now the hidden digit is the one parity depends on. |
| `encoding="digit values", optimizer="Adam", epochs=300` | 22% on test numbers, 36% on training numbers: memorizing. |
| `modulus=12, layers=2` | 99.4% on test numbers, above 90% by epoch 104 instead of 186. Mod 2 and mod 4 are perfect. |
| Hidden 7 in the hundreds place (session 2, Part 5) | 99% on ordinary test numbers. On the 525 numbers with a 7 in the hundreds place: 3% exactly right, 100% parity, 3% mod 3. |

**The hidden digit is the one to dwell on.** The score on hidden numbers is 3%, far below the 17% from guessing. The network is not confused, it is confidently and consistently wrong: asked about 4712 it says 4 (wrong) with 53%, and about 4012 it says 4 (right) with 100%. The switch for "7 in the hundreds place" was never on during training, so its connections never moved from their small starting values. Measured: 68% of the time the network answers as if that 7 were a 0, 3, 6 or 9 (as if it added nothing to the digit sum), 28% as if it were a 2, 5 or 8, and only 3% correctly. It still gets parity, because parity never depended on that digit. A mathematician who knows the digit-sum rule applies it to any digit in any place. The network learned a separate table of digit classes for each place, and a table has no entry for something it never saw. The ordinary test score of 99% gave no warning.

**Other moduli.** Every one of these is a weighted digit sum: n mod m is the sum of each digit times 10^k mod m. For 3 and 9 every weight is 1. For 11 the weights alternate 1, -1. For 7 they are 1, 3, 2, 6 (ones digit first), because 10 is 3 mod 7. Moduli that factor into pieces depending on few digits (10, 20, 15's mod 5 piece) are easy. How "nice" the rule looks to a person does not predict how hard it is for the network: 7 is learned faster than 9, and 11 is slowest.

**Base 6.** In base 6 the answer is the last digit, so the problem is as easy as parity was. The encoding did all the work. A good follow-up: in base 6, how would n mod 3 or n mod 4 behave?

**Weight decay.** It charges the network for large connection strengths. Parity is cheap, the mod 3 rule is expensive, and at strength 3 the network cannot afford the expensive one. It settles for partial credit.

The button panel (`lab.panel()`, at the end of the notebook) still has all nine presets A to I for anyone who prefers clicking.
