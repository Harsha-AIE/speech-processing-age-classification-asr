"""
Fine-tuning HuBERT and Wav2Vec2-XLS-R-53 for Hindi ASR (Common Voice 27.0) with CTC.
Minimal, 4 GB-VRAM-safe, manual PyTorch loop.

Pipeline:  Audio -> Processor -> Encoder -> Linear CTC head -> CTC loss -> backprop -> WER/CER
CTC loss used: torch.nn.CTCLoss, computed EXPLICITLY in ctc_loss_fn() from the model logits
(the model is called without labels, so its built-in loss is NOT used).
"""
import gc
import json
import math
import random
import time
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModelForCTC, get_linear_schedule_with_warmup

# =============================== CONFIG ===============================
DATA_ROOT = r"E:\sem 7\Speech Processing\assign 3"          # searched recursively for train.tsv
OUTPUT_DIR = r"E:\sem 7\Speech Processing\assign 3\outputs"  # fine-tuned checkpoints go here
RESULTS_CSV = "results.csv"
N_TRAIN, N_VAL, N_TEST = 2000, 100, 300
EPOCHS = 8
GRAD_ACCUM = 8
LR = 5e-5          # was 1e-5
HEAD_LR = 1e-3     # keep

MIN_SEC, MAX_SEC = 1.0, 8.0              # skip very short / long clips (saves VRAM)

BATCH_SIZE = 1                           # loop is written for batch size 1 (no padding needed)
WARMUP_FRAC = 0.1
SEED = 42
SAVE_MODELS = True

# (name, HF checkpoint, normalize waveform?, number of top encoder layers to train)
#  - HuBERT-base has 12 layers  -> train all of them
#  - XLS-R-53 has 24 layers     -> train only the top 6 (4 GB VRAM)
#  In both cases the CNN feature encoder is frozen; the CTC head is always trained.
MODELS = [
    ("HuBERT", "facebook/hubert-base-ls960", False, 12),
    ("Wav2Vec2-XLS-R-53", "facebook/wav2vec2-large-xlsr-53", True, 6),
]
# ======================================================================

SR = 16000
PAD, UNK, WORD_DELIM = "[PAD]", "[UNK]", "|"
PAD_ID = 0  # also used as the CTC blank
CTC = torch.nn.CTCLoss(blank=PAD_ID, reduction="mean", zero_infinity=True)


# ------------------------------ text ---------------------------------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s))
    # drop punctuation (P*), symbols (S*) and zero-width format chars (Cf); keep Hindi letters/matras
    s = "".join(c for c in s if not unicodedata.category(c).startswith(("P", "S", "Cf")))
    return " ".join(s.split())


from collections import Counter

def build_vocab(texts, min_count=5):
    cnt = Counter(c for t in texts for c in t if c != " ")
    chars = sorted(c for c, n in cnt.items() if n >= min_count)
    tokens = [PAD, UNK, WORD_DELIM] + chars
    return {t: i for i, t in enumerate(tokens)}


def encode(text, vocab):
    return [vocab.get(WORD_DELIM if c == " " else c, vocab[UNK]) for c in text]


def decode(ids, id2tok):
    out, prev = [], None
    for i in ids:                       # greedy CTC decoding: collapse repeats, drop blank
        if i != prev and i != PAD_ID:
            out.append(id2tok[i])
        prev = i
    text = "".join(t for t in out if t != UNK).replace(WORD_DELIM, " ")
    return " ".join(text.split())


# ------------------------------ data ---------------------------------
def find_dataset(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    hits = sorted(root.rglob("train.tsv"))
    if not hits:
        raise FileNotFoundError(f"No train.tsv found under {root}. Extract the .tar.gz first.")
    lang_dir = hits[0].parent
    clips = lang_dir / "clips"
    if not clips.is_dir():
        alt = sorted(root.rglob("clips"))
        if not alt:
            raise FileNotFoundError(f"No 'clips' folder found under {root}")
        clips = alt[0]
    for s in ("train", "dev", "test"):
        if not (lang_dir / f"{s}.tsv").is_file():
            raise FileNotFoundError(f"Missing {lang_dir / (s + '.tsv')}")
    return lang_dir, clips


def read_tsv(tsv):
    # Common Voice TSVs contain stray quotes -> quoting=3 (QUOTE_NONE)
    df = pd.read_csv(tsv, sep="\t", quoting=3, dtype=str, usecols=["path", "sentence"]).dropna()
    df["text"] = df["sentence"].map(normalize_text)
    df = df[df["text"].str.len() > 0]
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def load_audio(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)  # MP3 supported by soundfile>=0.12
    wav = wav.mean(axis=1)                                          # mono
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)       # 16 kHz
    return wav.astype(np.float32)


def build_subset(df, clips, n, split):
    data, skipped, last_err = [], 0, None
    for path, text in zip(df["path"], df["text"]):
        if len(data) >= n:
            break
        f = clips / path
        if not f.is_file():
            skipped += 1
            continue
        try:
            wav = load_audio(f)
        except Exception as e:  # corrupt / undecodable mp3
            skipped += 1
            last_err = e
            continue
        if not (MIN_SEC <= len(wav) / SR <= MAX_SEC):
            skipped += 1
            continue
        data.append((wav, text))
    print(f"[data] {split}: kept {len(data)} clips (skipped {skipped})")
    if not data:
        raise RuntimeError(f"No usable audio for '{split}'. Last error: {last_err}. "
                           "If MP3 decoding failed, run: pip install -U soundfile (or install ffmpeg).")
    return data


def to_input(wav, normalize, device):
    x = torch.from_numpy(wav)
    if normalize:  # zero-mean / unit-variance (XLS-R expects this, HuBERT-base does not)
        x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
    return x.unsqueeze(0).to(device)


# ------------------------------ model --------------------------------
def ctc_loss_fn(logits, ids):
    """CTC LOSS: logits (1,T,V) from the linear CTC head -> log-probs (T,1,V) -> nn.CTCLoss."""
    log_probs = logits.float().log_softmax(-1).transpose(0, 1)
    targets = torch.tensor([ids], dtype=torch.long, device=logits.device)
    return CTC(log_probs, targets, (logits.size(1),), (len(ids),))


def set_trainable(model, top_k):
    base = model.base_model                      # model.hubert / model.wav2vec2
    layers = base.encoder.layers
    if top_k >= len(layers):
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        for layer in layers[-top_k:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in base.encoder.layer_norm.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():     # linear CTC head
            p.requires_grad = True
    model.freeze_feature_encoder()               # CNN feature extractor always frozen


# ------------------------------ metrics ------------------------------
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(refs, hyps, unit):
    errs = total = 0
    for r, h in zip(refs, hyps):
        r_, h_ = (r.split(), h.split()) if unit == "word" else (list(r), list(h))
        errs += edit_distance(r_, h_)
        total += len(r_)
    return 100.0 * errs / max(total, 1)


# --------------------------- train / evaluate ------------------------
@torch.no_grad()
def predict(model, data, normalize, device, use_amp, id2tok):
    model.eval()
    refs, hyps = [], []
    for wav, text in data:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_values=to_input(wav, normalize, device)).logits
        hyps.append(decode(logits.argmax(-1)[0].cpu().tolist(), id2tok))
        refs.append(text)
        del logits
    return refs, hyps

@torch.no_grad()
def val_loss(model, data, vocab, normalize, device, use_amp):
    model.eval()
    tot = 0.0
    for wav, text in data:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_values=to_input(wav, normalize, device)).logits
        tot += ctc_loss_fn(logits, encode(text, vocab)).item()
    return tot / len(data)

def run_model(name, hf_id, normalize, top_k, vocab, train, val, test, device):
    print(f"\n{'=' * 60}\nFine-tuning {name}  ({hf_id})\n{'=' * 60}")
    use_amp = device.type == "cuda"
    id2tok = {i: t for t, i in vocab.items()}

    # Encoder + freshly initialised Linear CTC head (lm_head) sized to our Hindi vocab
    model = AutoModelForCTC.from_pretrained(
        hf_id, vocab_size=len(vocab), pad_token_id=PAD_ID, ctc_zero_infinity=True,
        layerdrop=0.0, mask_time_prob=0.05, ignore_mismatched_sizes=True,
    ).to(device)
    set_trainable(model, top_k)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable params: {n_tr / 1e6:.1f}M / {n_all / 1e6:.1f}M "
          f"(top {top_k} encoder layers + CTC head; CNN feature encoder frozen)")

    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("lm_head")]
    enc = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("lm_head")]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR}, {"params": head, "lr": HEAD_LR}], weight_decay=0.0)
    total_updates = max(1, math.ceil(len(train) * EPOCHS / GRAD_ACCUM))
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * total_updates), total_updates)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        run_loss, n_seen, pending, oom = 0.0, 0, 0, 0
        for i, idx in enumerate(order, 1):
            wav, text = train[idx]
            ids = encode(text, vocab)
            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(input_values=to_input(wav, normalize, device)).logits  # encoder + CTC head
                loss = ctc_loss_fn(logits, ids)                                            # <-- CTC LOSS
                if torch.isfinite(loss):
                    scaler.scale(loss / GRAD_ACCUM).backward()                             # <-- backprop
                    run_loss += loss.item()
                    n_seen += 1
                    pending += 1
                del logits, loss
            except torch.cuda.OutOfMemoryError:
                oom += 1
                torch.cuda.empty_cache()
            if (i % GRAD_ACCUM == 0 or i == len(order)) and pending > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
            if i % 50 == 0 or i == len(order):
                print(f"[train] epoch {epoch}/{EPOCHS}  sample {i}/{len(order)}  "
                      f"avg CTC loss {run_loss / max(n_seen, 1):.3f}  ({time.time() - t0:.0f}s)"
                      + (f"  OOM-skipped {oom}" if oom else ""))
                run_loss, n_seen = 0.0, 0
        print(f"[val] epoch {epoch}  CTC loss {val_loss(model, val, vocab, normalize, device, use_amp):.3f}")

    if SAVE_MODELS:
        out = Path(OUTPUT_DIR) / name
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[save] checkpoint -> {out}")

    refs, hyps = predict(model, test, normalize, device, use_amp, id2tok)
    wer, cer = error_rate(refs, hyps, "word"), error_rate(refs, hyps, "char")
    print(f"\n{'=' * 30}\n{name} RESULTS\n{'=' * 30}\nWER: {wer:.2f}\nCER: {cer:.2f}\n")
    for r, h in list(zip(refs, hyps))[:5]:
        print(f"Reference:  {r}\nPrediction: {h}\n")
    return {"Model": name, "WER": round(wer, 2), "CER": round(cer, 2)}


# -------------------------------- main -------------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch {torch.__version__} | torch CUDA build: {torch.version.cuda} | device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[env] GPU: {p.name} ({p.total_memory / 1024 ** 3:.1f} GB) | mixed precision (fp16): ON")
    else:
        print("[env] CUDA not available -> running on CPU (slow). Install a CUDA build of PyTorch.")

    lang_dir, clips = find_dataset(DATA_ROOT)
    print(f"[data] found dataset: {lang_dir}\n[data] clips folder: {clips}")

    train_df, val_df, test_df = (read_tsv(lang_dir / f"{s}.tsv") for s in ("train", "dev", "test"))
    print(f"[data] full split sizes  train={len(train_df)}  dev={len(val_df)}  test={len(test_df)}")
    vocab = build_vocab(train_df["text"])
    print(f"[data] vocab size: {len(vocab)} (incl. {PAD}, {UNK}, '{WORD_DELIM}')")

    train = build_subset(train_df, clips, N_TRAIN, "train")
    val = build_subset(val_df, clips, N_VAL, "val")
    test = build_subset(test_df, clips, N_TEST, "test")
    print(f"[data] subset sizes  train={len(train)}  val={len(val)}  test={len(test)}")

    results = []
    for name, hf_id, normalize, top_k in MODELS:
        results.append(run_model(name, hf_id, normalize, top_k, vocab, train, val, test, device))
        gc.collect()                       # model is out of scope -> free it before loading the next
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'Model':<22}{'WER':>8}{'CER':>10}\n{'-' * 40}")
    for r in results:
        print(f"{r['Model']:<22}{r['WER']:>8.2f}{r['CER']:>10.2f}")
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {RESULTS_CSV}")


if __name__ == "__main__":
    main()
