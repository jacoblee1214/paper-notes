"""Tokenizer vocabulary check for the Alpamayo 2 Edge checkpoint (CPU, stdlib + regex).

Reads a downloaded tokenizer.json as untrusted data (json only, no code execution) and reports:
- size and kind of the vocabulary and the added-token table
- whether candidate control strings exist (ids, special flag, source)
- how a byte-level BPE with the file's own merges and pre-tokenizer splits candidate strings

The tokenizer file itself is not written to the results. Only counts, ids and flags are saved.

Run: python3 -I analysis/extra_tokenizer_vocab.py [path/to/tokenizer.json]
Output: analysis/extra_tokenizer_vocab_results.json
"""
import hashlib
import json
import sys
from pathlib import Path

import regex

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = ROOT / "data" / "hf_tokenizer" / "tokenizer.json"
OUT = ROOT / "analysis" / "extra_tokenizer_vocab_results.json"

CANDIDATES = [
    "<NOMINAL>", "NOMINAL", "<nominal>", "<COC>", "<coc>", "<think>", "</think>",
    "<i0>", "<i1000>", "<i3999>", "<i4000>", "<|endoftext|>", "<|im_start|>", "<|im_end|>",
]
SAMPLE_STRINGS = ["<NOMINAL>", "<i0>", "<i1000>", "<think>"]
SELF_CHECK_STRINGS = ["Decelerate and turn right at the intersection.", "NOMINAL: keep going."]


def bytes_to_unicode():
    """GPT-2 style byte <-> printable-unicode map used by byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    return data, hashlib.sha256(raw).hexdigest(), len(raw)


def bpe(word, ranks):
    parts = list(word)
    if len(parts) < 2:
        return parts
    while True:
        best, best_rank = None, None
        for i in range(len(parts) - 1):
            r = ranks.get((parts[i], parts[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = i, r
        if best is None:
            return parts
        parts = parts[:best] + [parts[best] + parts[best + 1]] + parts[best + 2:]


def encode(text, added, vocab, ranks, pre_pattern, byte_map):
    """Split out added tokens first (longest match), then pre-tokenize and BPE the rest."""
    pieces = []  # (is_added, string)
    i = 0
    added_sorted = sorted(added, key=len, reverse=True)
    buf = ""
    while i < len(text):
        hit = next((a for a in added_sorted if text.startswith(a, i)), None)
        if hit is None:
            buf += text[i]
            i += 1
            continue
        if buf:
            pieces.append((False, buf))
            buf = ""
        pieces.append((True, hit))
        i += len(hit)
    if buf:
        pieces.append((False, buf))

    ids, toks = [], []
    for is_added, s in pieces:
        if is_added:
            ids.append(vocab[s])
            toks.append(s)
            continue
        # Isolated split: each regex match is one pre-token, unmatched gaps are kept as pre-tokens too.
        pos = 0
        for m in pre_pattern.finditer(s):
            if m.start() > pos:
                chunks = [s[pos:m.start()]]
            else:
                chunks = []
            chunks.append(m.group(0))
            for c in chunks:
                if c:
                    _emit(c, ids, toks, vocab, ranks, byte_map)
            pos = m.end()
        if pos < len(s):
            _emit(s[pos:], ids, toks, vocab, ranks, byte_map)
    return ids, toks


def _roundtrip(text, toks, added_by_content, inv_byte):
    """True when the BPE pieces, mapped back to bytes, rebuild the input text."""
    out = bytearray()
    for t in toks:
        if t in added_by_content:
            out += t.encode("utf-8")
        else:
            out += bytes(inv_byte[ch] for ch in t)
    return out.decode("utf-8", "replace") == text


def _emit(chunk, ids, toks, vocab, ranks, byte_map):
    mapped = "".join(byte_map[b] for b in chunk.encode("utf-8"))
    for t in bpe(mapped, ranks):
        ids.append(vocab[t])
        toks.append(t)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOKENIZER
    data, sha, nbytes = load(path)
    model = data["model"]
    vocab = model["vocab"]
    added = data["added_tokens"]
    added_by_content = {a["content"]: a for a in added}

    result = {
        "file": {
            "bytes": nbytes,
            "sha256": sha,
            "source": "https://huggingface.co/nvidia/Alpamayo2-Edge/resolve/main/tokenizer.json",
            "revision_x_repo_commit": "b92b6aa326bd3e2235db28f248ad5a30b5890e64",
        },
        "model_type": model.get("type"),
        "base_vocab_size": len(vocab),
        "merges": len(model.get("merges", [])),
        "added_tokens": len(added),
        "added_special": sum(1 for a in added if a.get("special")),
        "added_ids_min_max": [min(a["id"] for a in added), max(a["id"] for a in added)],
        "added_ids_overlap_base_vocab": sum(1 for a in added if a["content"] in vocab),
        "added_ids_out_of_base_range": sum(1 for a in added if a["id"] >= len(vocab)),
        "candidates": {},
    }

    for c in CANDIDATES:
        in_added = c in added_by_content
        in_vocab = c in vocab
        entry = {"in_added_tokens": in_added, "in_base_vocab": in_vocab}
        if in_added:
            entry["added_id"] = added_by_content[c]["id"]
            entry["added_special"] = bool(added_by_content[c].get("special"))
        if in_vocab:
            entry["base_id"] = vocab[c]
        result["candidates"][c] = entry

    # Substring search over all content strings (exact match is already above).
    nominal_like = sorted(
        {a["content"] for a in added if "nominal" in a["content"].lower()}
        | {t for t in vocab if "nominal" in t.lower()}
    )
    result["strings_containing_nominal_case_insensitive"] = nominal_like[:50]
    result["count_strings_containing_nominal"] = len(nominal_like)

    traj = [a for a in added if a["content"].startswith("<i") and a["content"].endswith(">")]
    result["added_trajectory_like_tokens"] = len(traj)
    result["trajectory_like_id_range"] = (
        [min(a["id"] for a in traj), max(a["id"] for a in traj)] if traj else None
    )

    byte_map = bytes_to_unicode()
    ranks = {}
    for r, m in enumerate(model.get("merges", [])):
        pair = tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
        ranks[pair] = r
    pre = data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    pre_pattern = regex.compile(pre)

    added_contents = list(added_by_content.keys())
    full_vocab = dict(vocab)
    full_vocab.update({a["content"]: a["id"] for a in added})
    inv_byte = {v: k for k, v in byte_map.items()}
    splits = {}
    for s in SAMPLE_STRINGS + SELF_CHECK_STRINGS:
        ids, toks = encode(s, added_contents, full_vocab, ranks, pre_pattern, byte_map)
        entry = {"n_tokens": len(ids), "ids": ids}
        if s in SAMPLE_STRINGS:
            entry["tokens"] = toks
            entry["tokens_are_added"] = [t in added_by_content for t in toks]
        else:
            # Round-trip check of the BPE implementation on ordinary English text.
            entry["roundtrip_ok"] = _roundtrip(s, toks, added_by_content, inv_byte)
        splits[s] = entry
    result["sample_splits"] = splits

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("base_vocab_size", "added_tokens", "added_special",
                                             "count_strings_containing_nominal")}, ensure_ascii=False))
    for c in CANDIDATES:
        print(c, result["candidates"][c])
    for s in SAMPLE_STRINGS:
        print(s, splits[s]["n_tokens"], splits[s]["ids"])


if __name__ == "__main__":
    main()
