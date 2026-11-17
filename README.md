# Training logs and final FastPitch checkpoints - PBL Praktikum 2, Topik 7

- Course: CSGE603130 Pemrosesan Bahasa Lisan, Fasilkom UI, Gasal 2026/2027 (PBL Praktikum 2, TTS)
- Topic 7: Pitch Statistics & Gender, Kelompok G
- Student: Rifqisyandi Khairurrizal (NPM 2306152216), working solo

Research question: FastPitch's config uses `pitch_mean=152.3`, `pitch_std=64.0` (LJSpeech, female English).
How much does keeping those values on another speaker hurt (MCD), and is the loss different for a female vs a male speaker?

This repo holds, per fine-tuned run, the latest training log folder and the final `.nemo` (FastPitch). Metrics are in `hasil.csv`
inside the SCeLE submission zip `Praktikum2_2306152216.zip` (not duplicated here).

## Runs

All runs: FastPitch fine-tuned from NGC `tts_en_fastpitch`, NeMo 2.5.0, seed 42, max 100 epochs, batch 16, lr 2e-4 (Adam, no scheduler), Kaggle T4.

| run_id | speaker / data | pitch stats used (mean / std) | max epochs / selected epoch | files |
|---|---|---|---|---|
| P1-F-own | Javanese female 08736 (OpenSLR 41), 157 train utt | own: 288.72 / 53.74 | 100 / 85 (ckpt index 84) | `P1-F-own/` |
| P2-F-ljs | Javanese female 08736, same data | template default: 152.3 / 64.0 | 100 / 35 (index 34) | `P2-F-ljs/` |
| P3-M-own | Javanese male 05522 (OpenSLR 41), 157 train utt | own: 157.34 / 30.95 | 100 / 95 (index 94) | `P3-M-own/` |
| P4-M-ljs | Javanese male 05522, same data | template default: 152.3 / 64.0 | 100 / 15 (index 14) | `P4-M-ljs/` |
| ID-FP | Indonesian, student's own voice, 164 train sentences | own: 162.16 / 28.25 | 100 / 50 (index 49) | `ID-FP/` |

The pretrained `tts_en_fastpitch` itself was trained with 212.36 / 68.53, so "ljs" runs use the template default from the config.
Pitch stats were computed with `librosa.pyin(fmin=30, fmax=512, sr=22050)` on the train split, unvoiced frames dropped.

## Files (per run folder)

- `<run_id>.nemo` - final FastPitch checkpoint (Git LFS). It is the checkpoint with the best validation loss (`save_best_model=true`,
  validation every 5 epochs), i.e. the selected epoch above, not the last epoch.
- `FastPitch/<timestamp>/` - the (latest and only) NeMo experiment folder: `events.out.tfevents.*` (TensorBoard), `hparams.yaml`,
  `cmd-args.log`, `lightning_logs.txt`, `nemo_log_globalrank-0_localrank-0.txt`, `nemo_error_log.txt`.
- `status.json` - epoch / elapsed time / peak VRAM written by the training wrapper. Its `epoch` field is the wrapper's counter after the last epoch (101 for P1-P4, 100 for ID-FP); `max_epochs` was 100 for every run.
- `train_stdout.log` - full stdout of the training process.
- `kaggle_run_logs/v4.log`, `v5.log` - Kaggle execution logs of the two notebook runs. v4 trained P1-F-own..P4-M-ljs; v5 restored those four unchanged
  and added ID-FP. The P1-F-own..P4-M-ljs `.nemo` files and log folders of v5 are byte-identical to v4 (sha256 checked); the copies here come from v5.

HiFi-GAN was **not** trained: all synthesis used the pretrained NGC `tts_en_hifigan`, frozen, so there is no HiFi-GAN model in this repo.
No audio of any kind is included (no recordings, no synthesized samples).

Redacted: the Kaggle account name in the dataset path of `kaggle_run_logs/*.log` was replaced by `<kaggle-user>`. A scan for tokens, API keys,
e-mail addresses, local user paths and phone numbers found nothing else. (Container paths such as `/kaggle/working/...` are left as is.)

## Loading a checkpoint

```python
from nemo.collections.tts.models import FastPitchModel, HifiGanModel   # nemo_toolkit[tts]==2.5.0
fp = FastPitchModel.restore_from("P1-F-own/P1-F-own.nemo").eval()
hg = HifiGanModel.from_pretrained("tts_en_hifigan").eval()
tokens = fp.parse("your text")
mel = fp.generate_spectrogram(tokens=tokens)
audio = hg.convert_spectrogram_to_audio(spec=mel)
```

## Git LFS note

`*.nemo` files (187 MB each, ~0.94 GB total) are stored with Git LFS. GitHub Free includes 10 GiB LFS storage and 10 GiB/month bandwidth
(assumption, not checked against current docs); each full download of the 5 models costs about 0.94 GB of the owner's monthly LFS bandwidth.
If that quota is exhausted, ask the owner for the models as GitHub Release assets instead.

## License and attribution

- Initialisation: NVIDIA NGC `tts_en_fastpitch` (and `tts_en_hifigan` for synthesis). To our knowledge these are published under CC-BY-4.0 on NGC; I could not verify
  the current license text, so check the NGC model card before reuse. The fine-tuned checkpoints are derivative works and carry the same attribution.
- Javanese data: OpenSLR SLR41 (Javanese multi-speaker TTS), CC BY-SA 4.0 (speakers 08736 and 05522). Audio not redistributed here.
- Indonesian data: the student's own recordings of sentences from the TVRI BISINDO corpus. The recordings are NOT included. ID-FP models the student's own voice.
- Course material and code structure: PBL Praktikum 2 template, Fasilkom UI.
