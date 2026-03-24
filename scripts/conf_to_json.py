#!/usr/bin/env python3
"""Convert .conf INI config to JSON for the chat binary."""
import sys, json

def parse_conf(path):
    sections = {}
    current = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                current = line[1:-1]
                sections[current] = {}
                continue
            for sep in ['\t', '=']:
                if sep in line:
                    key, val = line.split(sep, 1)
                    key = key.strip()
                    val = val.split('#')[0].strip()
                    sections[current][key] = val
                    break
    return sections

conf = parse_conf(sys.argv[1])
m = conf.get('model', {})
o = conf.get('optimization', {})
d = conf.get('data', {})

config = {
    "d_model": int(m.get('d_model', 256)),
    "vocab_size": int(m.get('vocab_size', 50257)),
    "n_layers": int(m.get('n_layers', 4)),
    "n_heads": int(m.get('n_heads', 8)),
    "n_kv_heads": int(m.get('n_kv_heads', -1)),
    "head_dim": int(m.get('head_dim', -1)),
    "rope_theta": int(m.get('rope_theta', 500000)),
    "layer_norm_eps": float(m.get('layer_norm_eps', 1e-6)),
    "init_std": float(m.get('init_std', 0.02)),
    "use_qk_norm": bool(int(m.get('use_qk_norm', 1))),
    "num_mtp_heads": int(m.get('num_mtp_heads', 0)),
    "mtp_loss_weight": float(m.get('mtp_loss_weight', 0.1)),
    "use_multi_res": bool(int(o.get('multi_res', 0))),
    "bpe_vocab_path": d.get('bpe_vocab', ''),
    "multi_res_char_buckets": 4096,
    "multi_res_phrase_buckets": 8192,
    "multi_res_inner_dim": 64,
}

out = sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout"
with open(out, 'w') as f:
    json.dump(config, f, indent=2)
    f.write('\n')
