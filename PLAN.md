# liltemp: pilot results and plan

## what the pilot found (5 oct 2026)

ran 9.6h overnight (stopped at 06:53 while scoring pythia-410m@16k round 2). all 14 models got round 1 (100 prompts), 8 got round 2 (200 prompts). 15 temps, 0.1 to 1.5. judge: qwen2.5-3b.

| model | params | own bpb | T* | 95% ci | n |
|---|---|---|---|---|---|
| smollm2-1.7b | 1711M | 0.746 | 0.827 | 0.807-0.842 | 100 |
| pythia-2.8b | 2775M | 0.766 | 0.815 | 0.792-0.833 | 100 |
| pythia-1.4b | 1415M | 0.810 | 0.778 | 0.750-0.806 | 100 |
| pythia-1b | 1012M | 0.856 | 0.764 | 0.746-0.783 | 200 |
| smollm2-360m | 362M | 0.878 | 0.736 | 0.721-0.751 | 200 |
| pythia-410m | 405M | 0.911 | 0.708 | 0.688-0.724 | 200 |
| pythia-1.4b@16k | 1415M | 0.912 | 0.700 | 0.674-0.723 | 100 |
| pythia-410m@64k | 405M | 0.932 | 0.706 | 0.682-0.725 | 100 |
| smollm2-135m | 135M | 0.970 | 0.673 | 0.656-0.695 | 200 |
| pythia-410m@16k | 405M | 0.982 | 0.653 | 0.628-0.677 | 100 |
| pythia-160m | 162M | 1.057 | 0.634 | 0.618-0.648 | 200 |
| lilbase | 297M | 1.062 | 0.626 | 0.607-0.639 | 200 |
| pythia-410m@4k | 405M | 1.121 | 0.635 | 0.619-0.650 | 200 |
| pythia-70m | 70M | 1.208 | 0.622 | 0.608-0.632 | 200 |

human text scores 0.766 bpb under the judge (call it h).

### the good news

1. **better models want a hotter temperature.** T* goes from 0.62 (pythia-70m) to 0.83 (smollm2-1.7b). every model's T* is under 1.0, and under the usual 0.7 default only for the weaker half.
2. **it's loss, not size.** pythia-1.4b@16k and pythia-410m have the same loss (0.912 vs 0.911) and the same T* (0.700 vs 0.708) with 3.5x different params. fitting on params alone: r2 0.68. adding params to a loss fit adds nothing (r2 0.896 to 0.899).
3. **best form so far: T* = 0.217 + 0.591 x (h / own_bpb)**, r2 0.955, leave-one-out error 0.014 on average, 0.042 worst. a plain linear fit on bpb (the one in analyze.py) is worse (r2 0.896) and bends at both ends. power law version: T* = 0.80 x (h / own_bpb)^0.67.
4. **it transfers across model families.** fit on pythia only, then predict the rest: smollm2-1.7b off by +0.006, smollm2-360m +0.003, smollm2-135m -0.014, lilbase -0.022. 3 of 4 land inside their own confidence interval.
5. one training run moving along the line: pythia-410m at 4k, 16k, 64k, 143k steps goes 0.635, 0.653, 0.706, 0.708.

### problems to fix before the real run

1. **the judge isn't better than the best test models.** smollm2-1.7b (0.746) is better than qwen2.5-3b (0.766) on these prompts, and pythia-2.8b ties it. the method assumes the judge knows real text better than the models it's judging. need a stronger judge.
2. **contamination.** wikitext-103 is old wikipedia, which is inside the pile (all pythia models) and probably in smollm2's data too. that makes own_bpb look better than it really is. need text written after every model's training cutoff.
3. **the second check doesn't work.** T* from distinct-word share is 0.90-0.93 for every model, so it can't tell them apart. 80 words is too short for that measure. drop it or swap it for something better.
4. **"optimal" is my own definition.** a reviewer will say T* is just where the numbers cross, not the temperature that actually gives the best text. need an outside check that text at T* is better than at 0.7 and 1.0.
5. **coarse grid.** every T* is between 0.6 and 0.85 but the grid spends most of its time on 0.1-0.5 and 1.1-1.5, and T* comes from straight lines between points 0.1 apart.
6. the big models only have 100 prompts, so their intervals are the widest.
7. the run stopped at 9.6h of a 13.4h deadline. no crash in the log, so check whether the mac slept or the process got killed.

## the real run

- **judge:** qwen2.5-7b in mlx 4-bit (about 4.5GB, fits fine in 16GB as long as no test model is loaded at the same time, which the script already does). keep qwen2.5-3b as a second judge on a subset so the paper can show the law doesn't depend on which judge you use.
- **prompts:** 500 passages from wikipedia articles created in 2026, so no model has seen them. same 40 word prompt + 80 word continuation setup.
- **temps:** 0.45 to 1.00 in steps of 0.05 (12 temps), plus 0.2 and 1.3 as anchors. finer where it matters, fewer in total.
- **models:** the same 14, plus more checkpoints of one run: pythia-160m and pythia-1b at steps 1k, 2k, 4k, 8k, 16k, 32k, 64k, 143k. same architecture, only loss changes, which is the cleanest test of the claim.
- **prompts per model:** 300 minimum, so the big models stop having the widest intervals.
- **compute:** the pilot averaged about 26 min per 100-prompt block. the real run is roughly 3 nights on the macbook. the pilot held 345 tok/s for the whole night, so throttling wasn't a problem, but keep it on a hard surface and plugged in.

## validating "optimal"

- **llm pairwise judge:** qwen2.5-7b-instruct (mlx 4-bit) sees two continuations of the same prompt, one at T* and one at 0.7 or 1.0, order randomised, and picks the more natural one. 200 pairs per model.
- **small human check:** 50 blind pairs rated by me (and a friend if possible). small, but it's the test reviewers trust most.
- if T* text doesn't win, that's a finding too. write it up honestly.

## analysis

- fit and compare: T* vs bpb, vs h/bpb, power law, vs log params, vs bpb + params. report r2 and leave-one-out error.
- headline test: fit on one family, predict the others.
- bootstrap over prompts for every T* interval (already in analyze.py).
- within-run curves for the pythia checkpoints.
- check judge robustness: same law with qwen 3b vs 7b.

## paper structure

1. **abstract.** most people sample at 0.7 or 1.0 whatever the model. we measure the temperature where a model's text is as surprising as human text, for 14+ models from 70M to 2.8B, and find it follows the model's loss, not its size, with one simple formula that predicts across model families.
2. **introduction.** small models say weird things at normal temperatures (lilchat example: 8/12 sane replies at 0.7, 12/12 at 0.4). no rule for picking T. our contribution: a measured law, the finding that loss beats size, and a free tool to compute it.
3. **related work.** sampling methods (top-k, nucleus, min-p, typical sampling), temperature and calibration, mauve and other text quality metrics, scaling laws, pythia checkpoints.
4. **method.** prompts and continuations, the judge, bits per byte, T* as the crossing point, bootstrap intervals, why the judge has to be stronger than the models.
5. **results.** fig 1: surprise vs temperature curves. fig 2: T* vs loss with the fit. table: all models. the loss vs size comparison. cross-family prediction. checkpoint trajectories.
6. **validation.** pairwise llm judge and human check.
7. **limitations.** one domain (wikipedia style text), base models only (no chat models), models under 3B, one judge family, pure temperature sampling with no top-p.
8. **conclusion.** the formula and how to use it: measure your model's loss on some fresh text, plug it in.
9. **appendix.** every curve, every interval, compute used, code link.

target: arxiv first, then TMLR (rolling, no deadline). an ICLR 2027 workshop is optional, only if it doesn't clash with TMLR review. full structure in paper/OUTLINE.md.

## order

1. finish the calculator paper first (rule: one paper at a time).
2. then: fresh prompts + mlx judge (1 weekend).
3. real run (3 nights).
4. validation (1 night + an afternoon for the human check).
5. analysis + figures (1 weekend).
6. write (2 weekends).
