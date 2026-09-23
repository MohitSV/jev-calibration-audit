# eve_patch

`eve_train_sft_mps.py` is a copy of `rlcd/train_sft.py` from
[anthony-maio/eve-rlcd](https://github.com/anthony-maio/eve-rlcd) at commit `57a179b7`.
It is patched only to run on Apple MPS instead of CUDA; `eve_train_sft_mps.diff` holds the
complete diff. The upstream code is MIT-licensed, and its license is reproduced in
`LICENSE-eve-rlcd`.

Run it with the eve-rlcd repository on `PYTHONPATH`. See `experiments/11-warmup-pilot.md`.
