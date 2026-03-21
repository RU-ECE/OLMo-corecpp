# Data Preparation Pipeline (100% C++)

All-C++ data pipeline. No Python. Fast parallel tokenization.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target prepare_data
```

## Usage

```bash
# From directory of .txt files (simple tokenizer)
./build/prepare_data --input data/texts/ --output data/tokens.npy

# GPT-2 BPE (parallel, fast)
./build/prepare_data --input data/texts/ --output data/tokens.npy \
  --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt --threads 8

# Download from HuggingFace (TinyStories for 3B training)
./build/prepare_data --download-hf roneneldan/TinyStories --output data/tokens.npy \
  --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt --max-tokens 100000000

# Download from URL (JSONL)
./build/prepare_data --download https://example.com/data.jsonl --output data/tokens.npy \
  --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt

# Random tokens for testing
./build/prepare_data --random 100000 --output data/sample_tokens.npy
```

## Download GPT-2 Tokenizer (one-time)

```bash
mkdir -p data/gpt2
curl -sL https://huggingface.co/gpt2/resolve/main/vocab.json -o data/gpt2/vocab.json
curl -sL https://huggingface.co/gpt2/resolve/main/merges.txt -o data/gpt2/merges.txt
```

## Options

| Option | Description |
|--------|-------------|
| `--input` | File or directory (.txt, .jsonl) |
| `--download` | URL to fetch (raw text or JSONL) |
| `--output` | Output .npy path |
| `--max-tokens` | Max tokens to save |
| `--vocab-file` | GPT-2 vocab.json (enables BPE) |
| `--merges-file` | GPT-2 merges.txt |
| `--threads` | Parallel threads (BPE only) |
| `--random <n>` | Generate n random tokens |
| `--vocab` | Save simple tokenizer vocab |
| `--load-vocab` | Load simple tokenizer vocab |

## Tokenizers

- **Simple**: Whitespace + punctuation. Builds vocab from corpus.
- **BPE (GPT-2)**: Use `--vocab-file` + `--merges-file`. Parallel, vocab_size=50257.

## Output

1D `.npy` array (uint16/uint32) compatible with `TokenDataset`.
