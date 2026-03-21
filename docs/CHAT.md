# Interactive Chat CLI

Chat with your trained OLMo model via the command line.

## 1. Train and save a checkpoint

Your previous run didn't save the model. Retrain with `--save`:

```bash
./build/olmo_train --train --data-path data/tinystories.npy --config configs/olmo2_125M.json \
  --device mps --batch-size 4 --seq-len 256 --steps 2000 --lr 3e-4 --warmup-steps 100 \
  --save checkpoints/125M.pt
```

## 2. Run the chat

```bash
./build/chat --checkpoint checkpoints/125M.pt --config configs/olmo2_125M.json \
  --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt
```

## Options

| Option | Description |
|--------|-------------|
| `--checkpoint` | Model checkpoint (.pt) |
| `--config` | Model config JSON |
| `--vocab-file` | GPT-2 vocab.json |
| `--merges-file` | GPT-2 merges.txt |
| `--device` | mps, cpu, or cuda |
| `--max-tokens` | Max tokens to generate (default: 128) |
| `--temperature` | Sampling temperature (default: 0.8) |

Type `quit` or `q` to exit.
