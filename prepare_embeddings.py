"""
prepare_embeddings.py

Both English fables (story N "The North Wind and the Sun" and story B) for all
three speakers (A, M, S) -> cuts of SEG_MS milliseconds with a fractional
OVERLAP between consecutive cuts -> ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
-> 192-dim embeddings, L2-normalised to unit length -> one CSV:

    data/embeddings_EN_EB_<seg>ms_overlap<pct>.csv
    columns: speaker, story, start_ms, dim_001 ... dim_192

Pipeline (same conventions as data_preparation.R and the project's overlap
experiments, only the cut length / stride differ):
  * mono mix, resample to 16 kHz
  * energy VAD on 25 ms frames / 10 ms hop: a frame is speech if within 40 dB
    of the recording's loudest frame
  * candidate cuts on a fixed grid with stride SEG_MS * (1 - OVERLAP)
    (e.g. 400 ms / 30% -> 280 ms stride, 120 ms shared; 750 ms / 0% -> 750 ms
    stride, no shared audio)
  * a cut is kept if >= 80% of its frames are speech
  * within each story the three speakers are balanced to the same number of
    cuts by evenly subsampling the longer ones

Usage (from the project folder):
    .venv/bin/python prepare_embeddings.py                      # 400 ms, 30% overlap
    .venv/bin/python prepare_embeddings.py --seg-ms 750 --overlap 0     # 750 ms, no overlap
    .venv/bin/python prepare_embeddings.py --audio-split 0.7            # split each recording in time

With --audio-split F each recording is split in TIME before cutting: the part
holding the first F of its speech frames is the training audio, the rest is the
test audio. Cuts are made separately inside each part (none crosses the
boundary, so no audio is shared between the sets), speakers are balanced
separately within each story x set, and the CSV gets a "set" column
(train/test) and the suffix _audiosplit<pct>.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)

REC_DIR = "recordings"
OUT_DIR = "data"
_ap = argparse.ArgumentParser()
_ap.add_argument("--seg-ms", type=int, default=400)
_ap.add_argument("--overlap", type=float, default=0.30)
_ap.add_argument("--audio-split", type=float, default=None)
_ap.add_argument("--force", action="store_true")
ARGS, _ = _ap.parse_known_args()

SEG_MS = ARGS.seg_ms
OVERLAP = ARGS.overlap
STRIDE_MS = SEG_MS * (1 - OVERLAP)          # 280 ms for the default 400 ms / 30%
AUDIO_SPLIT = ARGS.audio_split
OUT_CSV = os.path.join(OUT_DIR, f"embeddings_EN_EB_{SEG_MS}ms_overlap{round(100 * OVERLAP)}"
                       + (f"_audiosplit{round(100 * AUDIO_SPLIT)}" if AUDIO_SPLIT else "") + ".csv")
TARGET_SR = 16000
VAD_FLOOR_DB = 40.0
MIN_SPEECH_FRAC = 0.80

STORIES = {                                  # English, both fables
    "EN": {"A": "AEN", "M": "MEN", "S": "SEN"},
    "EB": {"A": "AEB", "M": "MEB", "S": "SEB"},
}

SEG_LEN = int(TARGET_SR * SEG_MS / 1000)     # 6400 samples at 400 ms
STRIDE = int(round(TARGET_SR * STRIDE_MS / 1000))   # 4480 samples at 280 ms
FR_LEN = int(TARGET_SR * 0.025)
FR_HOP = int(TARGET_SR * 0.010)


def load_16k_mono(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(wav.mean(axis=1))).unsqueeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).numpy()


def speech_frames(sig):
    """Energy VAD over the whole recording: True for 25 ms frames within 40 dB of the peak."""
    fr_idx = np.arange(0, len(sig) - FR_LEN + 1, FR_HOP)
    frames = sig[fr_idx[:, None] + np.arange(FR_LEN)[None, :]]
    fr_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
    return fr_db > (fr_db.max() - VAD_FLOOR_DB)


def cut_positions(sig, lo=0, hi=None, is_speech=None):
    """Kept cut starts on a fixed grid inside samples [lo, hi); cuts never cross hi."""
    n = len(sig) if hi is None else hi
    if is_speech is None:
        is_speech = speech_frames(sig)

    cand = np.arange(lo, n - SEG_LEN + 1, STRIDE)
    csum = np.concatenate([[0], np.cumsum(is_speech)])
    f1 = cand // FR_HOP                               # 0-based first frame
    f2 = (cand + SEG_LEN - FR_LEN) // FR_HOP          # 0-based last frame
    f2 = np.minimum(f2, len(is_speech) - 1)
    frac = (csum[f2 + 1] - csum[f1]) / (f2 - f1 + 1)
    return cand[frac >= MIN_SPEECH_FRAC], is_speech.mean()


def main():
    if os.path.exists(OUT_CSV) and not ARGS.force:
        print(f"{OUT_CSV} already exists (use --force to rebuild)")
        return

    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    os.makedirs(OUT_DIR, exist_ok=True)   # data/ is not tracked by git, so a fresh clone lacks it
    torch.set_num_threads(4)
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(OUT_DIR, "model_ecapa"),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )

    sets = ["train", "test"] if AUDIO_SPLIT else ["all"]
    rows = []
    for story, recs in STORIES.items():
        print(f"=== {story} ===")
        prep = {}                                     # (spk, set) -> (sig, starts)
        for spk, stem in recs.items():
            path = os.path.join(REC_DIR, stem + ".mpeg")
            sig = load_16k_mono(path)
            is_speech = speech_frames(sig)
            if AUDIO_SPLIT:
                ## boundary: first frame by which AUDIO_SPLIT of the speech frames have occurred
                b_frame = int(np.searchsorted(np.cumsum(is_speech), AUDIO_SPLIT * is_speech.sum()))
                b = b_frame * FR_HOP
                parts = {"train": cut_positions(sig, 0, b, is_speech)[0],
                         "test": cut_positions(sig, b, len(sig), is_speech)[0]}
                print(f"  {stem}.mpeg  {len(sig) / TARGET_SR:6.2f} s | speech {100 * is_speech.mean():4.1f}% "
                      f"| boundary at {b / TARGET_SR:6.2f} s | {len(parts['train'])} train + "
                      f"{len(parts['test'])} test cuts")
            else:
                parts = {"all": cut_positions(sig, 0, len(sig), is_speech)[0]}
                print(f"  {stem}.mpeg  {len(sig) / TARGET_SR:6.2f} s | speech {100 * is_speech.mean():4.1f}% "
                      f"| {len(parts['all'])} cuts")
            for st, starts in parts.items():
                prep[(spk, st)] = (sig, starts)

        for st in sets:
            n_cuts = min(len(prep[(spk, st)][1]) for spk in recs)
            print(f"  {st}: balanced to {n_cuts} cuts per speaker")
            for spk in recs:
                sig, starts = prep[(spk, st)]
                pick = np.round(np.linspace(0, len(starts) - 1, n_cuts)).astype(int)
                starts = starts[pick]
                segs = np.stack([sig[s:s + SEG_LEN] for s in starts]).astype(np.float32)
                with torch.no_grad():
                    emb = model.encode_batch(torch.from_numpy(segs)).squeeze(1).numpy()
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)     # L2 normalisation
                for s, e in zip(starts, emb):
                    rows.append((spk, story, st, int(round(1000 * s / TARGET_SR)), e))

    header = ["speaker", "story"] + (["set"] if AUDIO_SPLIT else []) + ["start_ms"] \
        + [f"dim_{i:03d}" for i in range(1, 193)]
    with open(OUT_CSV, "w") as f:
        f.write(",".join(f'"{h}"' for h in header) + "\n")
        for spk, story, st, start_ms, e in rows:
            set_col = f'"{st}",' if AUDIO_SPLIT else ""
            f.write(f'"{spk}","{story}",{set_col}{start_ms},' + ",".join(f"{v:.9g}" for v in e) + "\n")
    print(f"wrote {OUT_CSV}: {len(rows)} rows x {len(header)} cols")


if __name__ == "__main__":
    main()
