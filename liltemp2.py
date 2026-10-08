import gc
import json
import os
import time
import traceback
from pathlib import Path

import mlx.core as mx
import torch
import torch.nn.functional as F
from mlx_lm import load as mlx_load
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

HOURS = float(os.environ.get("HOURS", 13))
OUT = Path(os.environ.get("OUT", "runs2"))
CHUNK = int(os.environ.get("CHUNK", 100))
CONT_WORDS = 80
SPLIT_WORD = 40
MAX_NEW = 160
# T* for every pilot model sat between 0.6 and 0.85, and the 7b judge is the slow part, so only judge where it matters
TEMPS = [round(0.45 + 0.05 * i, 2) for i in range(12)]
# temperatures for the model's own loss on human text: gives the log-loss-optimal T and the self-consistent T
TCAL = [round(0.4 + 0.05 * i, 2) for i in range(25)]
REF = os.environ.get("REF", "mlx-community/Qwen2.5-7B-4bit")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# (name, repo, revision). order = priority within each round
MODELS = [
    ("pythia-70m", "EleutherAI/pythia-70m", None),
    ("pythia-160m", "EleutherAI/pythia-160m", None),
    ("pythia-410m", "EleutherAI/pythia-410m", None),
    ("pythia-1b", "EleutherAI/pythia-1b", None),
    ("pythia-1.4b", "EleutherAI/pythia-1.4b", None),
    ("pythia-2.8b", "EleutherAI/pythia-2.8b", None),
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", None),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", None),
    ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B", None),
    ("lilbase", "navthings/lilbase", None),
    ("pythia-410m@4k", "EleutherAI/pythia-410m", "step4000"),
    ("pythia-410m@16k", "EleutherAI/pythia-410m", "step16000"),
    ("pythia-410m@64k", "EleutherAI/pythia-410m", "step64000"),
    ("pythia-1.4b@16k", "EleutherAI/pythia-1.4b", "step16000"),
    ("pythia-160m@1k", "EleutherAI/pythia-160m", "step1000"),
    ("pythia-160m@2k", "EleutherAI/pythia-160m", "step2000"),
    ("pythia-160m@4k", "EleutherAI/pythia-160m", "step4000"),
    ("pythia-160m@8k", "EleutherAI/pythia-160m", "step8000"),
    ("pythia-160m@16k", "EleutherAI/pythia-160m", "step16000"),
    ("pythia-160m@32k", "EleutherAI/pythia-160m", "step32000"),
    ("pythia-160m@64k", "EleutherAI/pythia-160m", "step64000"),
    ("pythia-1b@4k", "EleutherAI/pythia-1b", "step4000"),
    ("pythia-1b@16k", "EleutherAI/pythia-1b", "step16000"),
]

if os.environ.get("ONLY"):
    MODELS = [m for m in MODELS if m[0] in os.environ["ONLY"].split(",")]

START = time.time()


def log(msg: str):
    el = time.time() - START
    print(f"[{time.strftime('%H:%M:%S')} +{el / 3600:4.1f}h] {msg}", flush=True)


def split_at(text: str, n_words: int) -> int:
    # character index where word n_words starts in text (text starts with a space)
    seen = 0
    for i, ch in enumerate(text):
        if ch != " " and (i == 0 or text[i - 1] == " "):
            if seen == n_words:
                return i
            seen += 1
    return len(text)


def rep4(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


# pythia breaks in bf16 (it was trained in fp16), so test models run fp32, fp16 for 2.8b to fit in 16GB
def load(repo: str, rev: str | None):
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    assert tok.is_fast, f"{repo} has no fast tokenizer, offsets won't work"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if repo == "EleutherAI/pythia-2.8b" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype).to(DEVICE).eval()
    return model, tok, sum(p.numel() for p in model.parameters())


def free():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    mx.clear_cache()


@torch.no_grad()
def generate(model, tok, prompts: list[str], temp: float, bs: int, seed: int) -> list[str]:
    tok.padding_side = "left"
    cfg = GenerationConfig(do_sample=True, temperature=temp, top_k=0, top_p=1.0, min_p=None, repetition_penalty=1.0,
                           max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEVICE)
        torch.manual_seed(seed + i)
        ids = model.generate(**enc, generation_config=cfg)
        out += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


# nats the test model spends on each continuation at each temperature in temps, plus its byte count.
# the vocab projection only runs on continuation positions, full logits for every position blow up memory
@torch.no_grad()
def own_nll(model, tok, pairs: list[tuple[str, str]], bs: int, temps) -> list[tuple[dict, int]]:
    tok.padding_side = "right"
    head = model.get_output_embeddings()
    order = sorted(range(len(pairs)), key=lambda k: len(pairs[k][0]) + len(pairs[k][1]))
    out = {}
    for i in range(0, len(pairs), bs):
        ids = order[i:i + bs]
        batch = [pairs[k] for k in ids]
        enc = tok([c + x for c, x in batch], return_tensors="pt", padding=True, return_offsets_mapping=True)
        starts = enc.pop("offset_mapping")[:, 1:, 0]
        mask = enc["attention_mask"][:, 1:].bool()
        enc = enc.to(DEVICE)
        hidden = model.base_model(**enc).last_hidden_state
        tgt = enc["input_ids"][:, 1:]
        for j, (c, x) in enumerate(batch):
            keep = ((starts[j] >= len(c)) & mask[j]).nonzero().squeeze(1)
            if not x.strip() or len(keep) == 0:
                out[ids[j]] = ({t: 0.0 for t in temps}, 0)
                continue
            keep = keep.to(DEVICE)
            logits = head(hidden[j, keep]).float()
            nll = {t: F.cross_entropy(logits / t, tgt[j, keep], reduction="sum").item() for t in temps}
            out[ids[j]] = (nll, len(x.encode()))
        del hidden
        if DEVICE == "mps" and (i // bs) % 20 == 19:
            torch.mps.empty_cache()
    return [out[k] for k in range(len(pairs))]


# the judge, in mlx: nats on the first 40 words and the last 40 words of each continuation, plus byte counts
def judge_nll(model, tok, pairs: list[tuple[str, str]], bs: int = 16) -> list[dict]:
    hf = tok._tokenizer
    pad = hf.pad_token_id if hf.pad_token_id is not None else hf.eos_token_id
    lm = model.model
    head = model.lm_head if hasattr(model, "lm_head") else lm.embed_tokens.as_linear
    order = sorted(range(len(pairs)), key=lambda k: len(pairs[k][0]) + len(pairs[k][1]))
    out = {}
    for i in range(0, len(pairs), bs):
        if len(pairs) >= 1000 and i and i % (len(pairs) // 4 // bs * bs) == 0:
            log(f"  judged {i}/{len(pairs)}")
        ids = order[i:i + bs]
        batch = [pairs[k] for k in ids]
        enc = hf([c + x for c, x in batch], return_offsets_mapping=True)
        width = max(len(e) for e in enc["input_ids"])
        grid = mx.array([e + [pad] * (width - len(e)) for e in enc["input_ids"]])
        hidden = lm(grid)
        # gather every scored position in the batch, one head call and one sync per batch
        rows, tgts, seg, cuts = [], [], [], []
        for j, (c, x) in enumerate(batch):
            n = len(enc["input_ids"][j])
            cut = len(c) + split_at(x, SPLIT_WORD)
            cuts.append(cut - len(c))
            starts = [s for s, _ in enc["offset_mapping"][j][1:n]]
            for p in range(n - 1):
                if starts[p] >= len(c) and x.strip():
                    rows.append(j * width + p)
                    tgts.append(enc["input_ids"][j][p + 1])
                    seg.append(2 * j + (starts[p] >= cut))
        sums = [0.0] * (2 * len(batch))
        if rows:
            flat = hidden.reshape(-1, hidden.shape[-1])[mx.array(rows)]
            logits = head(flat).astype(mx.float32)
            tok_nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, mx.array(tgts)[:, None], axis=-1)[:, 0]
            onehot = mx.array(seg)[:, None] == mx.arange(2 * len(batch))[None, :]
            sums = mx.sum(mx.where(onehot, tok_nll[:, None], 0.0), axis=0).tolist()
        for j, (c, x) in enumerate(batch):
            a, b, ca = sums[2 * j], sums[2 * j + 1], cuts[j]
            out[ids[j]] = {"nll": a + b, "bytes": len(x.encode()) if x.strip() else 0, "nll_a": a,
                           "bytes_a": len(x[:ca].encode()), "nll_b": b, "bytes_b": len(x[ca:].encode())}
        del hidden
    return [out[k] for k in range(len(pairs))]


# generation memory (the kv cache) grows with batch x layers x width, and 410m at batch 100 in fp32 swapped the mac
def gen_bs(n: int) -> int:
    return 100 if n < 2e8 else 32 if n < 6e8 else 16 if n < 2e9 else 8


def append(name: str, rows: list[dict]):
    with open(OUT / name, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read(name: str) -> list[dict]:
    p = OUT / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def run_block(mi: int, name: str, repo: str, rev: str | None, chunk: int, prompts, human_done: set):
    idxs = list(range(chunk * CHUNK, (chunk + 1) * CHUNK))
    pairs = [(prompts[i][0], prompts[i][1]) for i in idxs]
    model, tok, n = load(repo, rev)
    bs = gen_bs(n)
    log(f"{name} ({n / 1e6:.0f}M) chunk {chunk}: loaded, batch {bs}")

    own = own_nll(model, tok, pairs, max(8, bs // 4), TCAL)
    own_rows = [{"model": name, "params": n, "idx": i, "nll": a[1.0], "nll_t": a, "bytes": b}
                for i, (a, b) in zip(idxs, own)]

    gens = []
    for temp in TEMPS:
        t0 = time.time()
        texts = generate(model, tok, [p for p, _ in pairs], temp, bs, seed=chunk * 100000 + mi * 1000 + int(temp * 100))
        rows = []
        for i, t in zip(idxs, texts):
            w = t.split()[:CONT_WORDS]
            rows.append({"model": name, "temp": temp, "idx": i, "text": (" " + " ".join(w)) if w else "", "n_words": len(w)})
        # the model's own surprise at its own samples, at the same temperature it sampled them (entropy calibration)
        selfs = own_nll(model, tok, [(prompts[r["idx"]][0], r["text"]) for r in rows], max(8, bs // 4), [temp])
        for r, (a, b) in zip(rows, selfs):
            r.update(self_nll=a[temp], self_bytes=b)
        gens += rows
        log(f"  {name} T={temp:.2f}  {len(texts) * MAX_NEW / (time.time() - t0):6.0f} tok/s")
        free()
    del model, tok
    free()

    ref, rtok = mlx_load(REF)
    log(f"  judging {len(gens)} texts with {REF}")
    scored = judge_nll(ref, rtok, [(prompts[g["idx"]][0], g["text"]) for g in gens])
    for g, s in zip(gens, scored):
        g.update({f"ref_{k}": v for k, v in s.items()}, rep4=rep4(g["text"]))
    if chunk not in human_done:
        hs = judge_nll(ref, rtok, pairs)
        append("human.jsonl", [{"idx": i, **{f"ref_{k}": v for k, v in s.items()}, "rep4": rep4(c)}
                               for i, s, (_, c) in zip(idxs, hs, pairs)])
        human_done.add(chunk)
    del ref, rtok
    free()

    append("gens.jsonl", gens)
    append("own.jsonl", own_rows)
    append("done.jsonl", [{"model": name, "chunk": chunk}])


def main():
    OUT.mkdir(exist_ok=True)
    path = OUT / "prompts.json"
    if not path.exists():
        raise SystemExit(f"no {path}, run: python fresh.py {OUT}")
    prompts = json.loads(path.read_text())
    done = {(d["model"], d["chunk"]) for d in read("done.jsonl")}
    human_done = {h["idx"] // CHUNK for h in read("human.jsonl")}
    deadline = START + HOURS * 3600
    took, broken = {}, set()
    log(f"device {DEVICE}, {len(MODELS)} models, {len(TEMPS)} temps, judge {REF}, deadline in {HOURS}h, {len(done)} blocks already done")

    for chunk in range(len(prompts) // CHUNK):
        for mi, (name, repo, rev) in enumerate(MODELS):
            if (name, chunk) in done or name in broken:
                continue
            if time.time() + took.get(name, 0) > deadline:
                log(f"stopping: {name} chunk {chunk} won't finish before the deadline")
                return
            t0 = time.time()
            try:
                run_block(mi, name, repo, rev, chunk, prompts, human_done)
            except Exception:
                log(f"{name} failed, skipping it for the rest of the run:\n{traceback.format_exc()}")
                broken.add(name)
                free()
                continue
            took[name] = time.time() - t0
            log(f"{name} chunk {chunk} done in {took[name] / 60:.1f} min")
        log(f"=== round {chunk} finished ({(chunk + 1) * CHUNK} prompts per point) ===")
    log("all chunks done")


if __name__ == "__main__":
    main()
