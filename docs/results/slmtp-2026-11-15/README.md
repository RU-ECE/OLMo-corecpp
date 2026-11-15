# SL-MTP raw results (OLMo-2-0425-1B frozen, M4 Pro)

The numbers behind docs/SPEED_RESULTS.md section 2 ("Training the heads for drafting").

| File | What |
|---|---|
| results.jsonl, saved_run.json | the 2026-11-14 kill test (rank 32/128/512, full rank, a pinned 0) |
| selfdistill.jsonl | 4 epochs, real next token vs the model's own picks (`--target greedy`) |
| scale_train.json | self-distilled heads on 1.6M tokens |
| soft_train.json | `--target soft` (the model's whole distribution) |
| decode.jsonl | decoding, prompts from stories past 2,000 |
| decode_unseen.jsonl, decode_soft.jsonl | decoding, prompts from stories past 20,000 (never trained on) |
| *.sh | the scripts that produced them (run from runs/killtest) |

The heads themselves (`heads_sl_r128*.pt`, 6 MB each) stay in runs/killtest (gitignored).
