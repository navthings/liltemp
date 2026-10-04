import gc
import json
import os
import random
import re
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

HOURS = float(os.environ.get("HOURS", 15.5))
OUT = Path(os.environ.get("OUT", "runs"))
CHUNK = int(os.environ.get("CHUNK", 100))
MAX_CHUNKS = 10
PROMPT_WORDS = 40
CONT_WORDS = 80
MAX_NEW = 160
TEMPS = [round(0.1 * i, 1) for i in range(1, 16)][::int(os.environ.get("TEMP_STEP", 1))]
REF = os.environ.get("REF", "Qwen/Qwen2.5-3B")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# (name, repo, revision). order = priority, the first round over all of these is the pilot
MODELS = [
    ("pythia-70m", "EleutherAI/pythia-70m", None),
    ("pythia-160m", "EleutherAI/pythia-160m", None),
    ("pythia-410m", "EleutherAI/pythia-410m", None),
    ("pythia-1b", "EleutherAI/pythia-1b", None),
    ("lilbase", "navthings/lilbase", None),
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", None),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", None),
    ("pythia-410m@4k", "EleutherAI/pythia-410m", "step4000"),
    ("pythia-410m@16k", "EleutherAI/pythia-410m", "step16000"),
    ("pythia-410m@64k", "EleutherAI/pythia-410m", "step64000"),
    ("pythia-1.4b", "EleutherAI/pythia-1.4b", None),
    ("pythia-1.4b@16k", "EleutherAI/pythia-1.4b", "step16000"),
    ("pythia-2.8b", "EleutherAI/pythia-2.8b", None),
    ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B", None),
]

if os.environ.get("ONLY"):
    MODELS = [m for m in MODELS if m[0] in os.environ["ONLY"].split(",")]

START = time.time()


def log(msg: str):
    el = time.time() - START
    print(f"[{time.strftime('%H:%M:%S')} +{el / 3600:4.1f}h] {msg}", flush=True)


def detok(s: str) -> str:
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([.,;:!?)\]])", r"\1", s)
    s = re.sub(r"([(\[]) ", r"\1", s)
    return s.replace(" n't", "n't").replace(" 's", "'s")


def get_prompts() -> list[tuple[str, str]]:
    path = OUT / "prompts.json"
    if path.exists():
        return [tuple(p) for p in json.loads(path.read_text())]
    need, rows = CHUNK * MAX_CHUNKS, []
    for split in ["test", "validation", "train"]:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        for line in ds["text"]:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            words = detok(line).split()
            if len(words) >= PROMPT_WORDS + CONT_WORDS:
                rows.append((" ".join(words[:PROMPT_WORDS]), " " + " ".join(words[PROMPT_WORDS:PROMPT_WORDS + CONT_WORDS])))
        if len(rows) >= need:
            break
    random.Random(0).shuffle(rows)
    rows = rows[:need]
    path.write_text(json.dumps(rows))
    log(f"built {len(rows)} prompts from wikitext-103")
    return rows


def rep4(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


# pythia breaks in bf16 (it was trained in fp16), so test models run fp32, fp16 for 2.8b to fit in 16GB
def dtype_for(repo: str):
    if repo == REF:
        return torch.bfloat16
    return torch.float16 if repo == "EleutherAI/pythia-2.8b" else torch.float32


def load(repo: str, rev: str | None):
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    assert tok.is_fast, f"{repo} has no fast tokenizer, offsets won't work"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype_for(repo)).to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters())
    return model, tok, n


def free():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def generate(model, tok, prompts: list[str], temp: float, bs: int, seed: int) -> list[str]:
    tok.padding_side = "left"
    cfg = GenerationConfig(do_sample=True, temperature=temp, top_k=0, top_p=1.0, min_p=None, repetition_penalty=1.0,
                           max_new_tokens=MAX_NEW,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEVICE)
        torch.manual_seed(seed + i)
        ids = model.generate(**enc, generation_config=cfg)
        out += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


# nats spent on the continuation given the context, plus its byte count (for bits per byte)
@torch.no_grad()
def cont_nll(model, tok, pairs: list[tuple[str, str]], bs: int) -> list[tuple[float, int]]:
    tok.padding_side = "right"
    res = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        enc = tok([c + x for c, x in batch], return_tensors="pt", padding=True, return_offsets_mapping=True)
        starts = enc.pop("offset_mapping")[:, 1:, 0]
        mask = enc["attention_mask"][:, 1:].bool()
        enc = enc.to(DEVICE)
        logits = model(**enc).logits
        tgt = enc["input_ids"][:, 1:]
        for j, (c, x) in enumerate(batch):
            if not x.strip():
                res.append((0.0, 0))
                continue
            nll = F.cross_entropy(logits[j, :-1].float(), tgt[j], reduction="none").cpu()
            keep = (starts[j] >= len(c)) & mask[j]
            res.append((nll[keep].sum().item(), len(x.encode())))
    return res


def gen_bs(n: int) -> int:
    return 100 if n < 5e8 else 50 if n < 2e9 else 25


def append(name: str, rows: list[dict]):
    with open(OUT / name, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read(name: str) -> list[dict]:
    p = OUT / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def run_block(mi: int, name: str, repo: str, rev: str | None, chunk: int, prompts, human_done: set):
    idxs = list(range(chunk * CHUNK, (chunk + 1) * CHUNK))
    pairs = [prompts[i] for i in idxs]
    model, tok, n = load(repo, rev)
    bs = gen_bs(n)
    log(f"{name} ({n / 1e6:.0f}M) chunk {chunk}: loaded, batch {bs}")

    own = cont_nll(model, tok, pairs, max(8, bs // 4))
    own_rows = [{"model": name, "params": n, "idx": i, "nll": a, "bytes": b} for i, (a, b) in zip(idxs, own)]

    gens = []
    for temp in TEMPS:
        t0 = time.time()
        texts = generate(model, tok, [p for p, _ in pairs], temp, bs, seed=chunk * 100000 + mi * 1000 + int(temp * 10))
        for i, t in zip(idxs, texts):
            w = t.split()[:CONT_WORDS]
            gens.append({"model": name, "temp": temp, "idx": i, "text": (" " + " ".join(w)) if w else "", "n_words": len(w)})
        dt = time.time() - t0
        log(f"  {name} T={temp:.1f}  {len(texts) * MAX_NEW / dt:6.0f} tok/s")
    del model, tok
    free()

    ref, rtok, _ = load(REF, None)
    scored = cont_nll(ref, rtok, [(prompts[g["idx"]][0], g["text"]) for g in gens], 8)
    for g, (a, b) in zip(gens, scored):
        g.update(ref_nll=a, ref_bytes=b, rep4=rep4(g["text"]))
    if chunk not in human_done:
        hs = cont_nll(ref, rtok, pairs, 8)
        append("human.jsonl", [{"idx": i, "ref_nll": a, "ref_bytes": b, "rep4": rep4(c)}
                               for i, (a, b), (_, c) in zip(idxs, hs, pairs)])
        human_done.add(chunk)
    del ref, rtok
    free()

    append("gens.jsonl", gens)
    append("own.jsonl", own_rows)
    append("done.jsonl", [{"model": name, "chunk": chunk}])


def main():
    OUT.mkdir(exist_ok=True)
    prompts = get_prompts()
    done = {(d["model"], d["chunk"]) for d in read("done.jsonl")}
    human_done = {h["idx"] // CHUNK for h in read("human.jsonl")}
    deadline = START + HOURS * 3600
    took, broken = {}, set()
    log(f"device {DEVICE}, {len(MODELS)} models, {len(TEMPS)} temps, deadline in {HOURS}h, {len(done)} blocks already done")

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
