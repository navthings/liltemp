# paper outline (TMLR)

working title: **model loss predicts the best sampling temperature**
(calmer than "a law". TMLR reviewers check that every claim is backed up, so the title shouldn't promise more than the data shows.)

## how TMLR is different from a conference

- **no deadline.** submit when it's ready, reviews start within weeks.
- **no novelty bar.** two questions decide it: are the claims backed by clear, convincing evidence, and would some people in the ML community find it interesting. this paper is a good fit for both.
- **no hard page limit**, but long papers get slower reviews. aim for ~12 pages of main text and put the rest in the appendix.
- **double blind.** anonymised paper: no name, no "my model lilbase", code linked through an anonymous repo (anonymous.4open.science). arxiv is fine as long as the TMLR pdf itself stays anonymous.
- **reviews go back and forth.** there's a discussion and revision phase, so reviewers can ask for an extra experiment and I can add it. decision: accept, accept with minor revisions, or reject.
- **broader impact statement** is only required if the work has a real risk of harm. this doesn't, but a short paragraph doesn't hurt.
- **reproducibility counts.** code, generations and results can go in as supplementary (up to 100MB). everything runs on one laptop, which makes this paper easy to reproduce. say so loudly.
- template: `tmlr.sty` + `tmlr.bst` from the TMLR style file repo on github.
- don't submit it anywhere else while it's under review.

## the main rule for writing it

every claim in the paper gets a figure, table or number right next to it. no "this shows that X generalises" unless a test shows exactly that. the biggest reason TMLR papers get rejected is claiming more than the evidence supports.

## claims and the evidence for each

| claim | evidence |
|---|---|
| T* varies a lot between models (0.6 to 0.85) | table 1, fig 3 |
| T* is predicted by loss | fit comparison table, r2 + leave-one-out error |
| size adds nothing once loss is known | matched pairs + regression with both, fig 4 |
| it holds inside one training run | checkpoint trajectories, fig 5 |
| it transfers across model families | fit on pythia, predict the rest, table 2 |
| it isn't an artifact of the judge | same result with a second judge, table 4 |
| T* text is actually better | pairwise llm judge + human pairs + mauve, section 6 |
| it isn't just calibration | T* vs calibration temperature, fig 6 |

anything I can't back up goes in limitations instead.

## structure (~12 pages main text)

### abstract
most people sample every model at 0.7 or 1.0. we measure the temperature T* where a model's text is as surprising as human text to a stronger judge, for N base models from 70M to 2.8B across three families and checkpoints from inside training. T* ranges from ~0.62 to ~0.83 and is predicted by the model's loss, not its size. a simple formula fitted on one family predicts the others within ~0.02. text sampled at T* is preferred over the defaults in blind comparisons (if the validation holds). all experiments ran on one laptop and all code and generations are released.

### 1. introduction (1 page)
- small models fall apart at the usual temperatures (chat model example: 8/12 sane replies at 0.7, 12/12 at 0.4, anonymised).
- lots of work on *how* to sample, almost none on *what temperature a model needs*.
- contributions:
  1. a measurement method for T* using a stronger judge and fresh text
  2. T* follows loss, not size, across 3 families and inside training runs
  3. a simple predictive formula that transfers across families
  4. validation that T* text is preferred
  5. code, all generations and results, reproducible on a laptop

### 2. related work (1 page)
- truncation sampling: top-k, nucleus, typical, eta, min-p
- mirostat: controls perplexity while decoding. difference: we predict one temperature from loss before decoding, and explain it
- task-specific temperature (code, reasoning)
- calibration and temperature scaling
- scaling laws
- text quality metrics (mauve)

### 3. method (2 pages)
- 3.1 data: 500+ passages from wikipedia articles created in 2026. 40 word prompt, 80 word continuation. why fresh text matters.
- 3.2 models: pythia 70m to 2.8b + checkpoints, smollm2 135m/360m/1.7b, our 297M model. base models only. precision choices (pythia breaks in bf16).
- 3.3 generation: pure temperature sampling, grid 0.45 to 1.00 in 0.05 steps plus anchors, fixed seeds.
- 3.4 judge: qwen2.5-7b, 4-bit. scores continuations in bits per byte. table showing it beats every test model on human text.
- 3.5 defining T*: where judged surprise crosses the human level, interpolated, 95% intervals from bootstrapping prompts.
- 3.6 model quality: the model's own bits per byte on the human continuations.
- fig 1: pipeline sketch.

### 4. results (3.5 pages)
- 4.1 surprise vs temperature curves (fig 2)
- 4.2 T* vs loss and the fitted formula. compare forms (linear in loss, in h/loss, power law, log params). table of r2 and leave-one-out error (fig 3, table 1)
- 4.3 loss, not size (fig 4: T* vs params next to T* vs loss, matched pairs, regression)
- 4.4 inside a training run (fig 5: pythia checkpoints)
- 4.5 across families (table 2)

### 5. robustness (1.5 pages)
TMLR has the room, so this lives in the main text, not just the appendix.
- second judge (qwen2.5-3b)
- min-p / top-p switched on
- longer continuations
- the old wikitext pilot vs fresh text (contamination check)
- table 4 with all of it

### 6. does T* give better text? (1.5 pages)
- pairwise llm judge, order randomised: T* vs 0.7, T* vs 1.0
- 50 blind human pairs
- mauve at T* vs the defaults
- report losses as well as wins

### 7. why does loss set the temperature? (1 page)
- toy model: a weaker model is a flattened copy of the truth (q ∝ p^s, s < 1), and sampling at T = s undoes the flattening. so worse loss, flatter model, lower T*.
- is it just calibration? compare T* with the temperature that minimises loss on human text (fig 6). keep the claims here modest, it's an explanation that fits, not a proof.

### 8. limitations (0.5 pages)
one domain, base models only, under 3B, one judge family, plain temperature sampling, "human-like" is defined by the judge, small human study.

### 9. conclusion (0.25 pages)
the formula and how to use it.

### broader impact (short)
better outputs from small models that run on phones and laptops. low risk.

### appendix
- A. every curve and interval
- B. wikitext pilot
- C. pythia in bf16
- D. judge quality vs every model
- E. compute: one 16GB macbook air, hours per model
- F. sample texts at T*, 0.7 and 1.0
- G. full robustness tables
- H. how to reproduce in one command

## what's required vs nice to have

**required before submitting** (without these the claims aren't backed up):
- fresh 2026 text
- a judge stronger than every test model
- the validation in section 6
- the second judge
- 300+ prompts per model

**nice to have** (makes it stronger, not needed to get in):
- chat models
- one 7B+ model on the free tpu
- more checkpoints
- a second domain like code

## order
1. finish the calculator paper (one paper at a time)
2. fresh prompts + mlx judge
3. full run (~3 nights)
4. validation
5. analysis + figures
6. write, put it on arxiv, submit to TMLR
7. reviewers ask for things, add them during the discussion phase
