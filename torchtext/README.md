# torchtext (vocab only)

A minimal, standalone reimplementation of `torchtext.vocab.Vocab` (and its two
factory functions and `VocabTransform`), extracted from
[pytorch/text](https://github.com/pytorch/text) v0.18.0 and rebuilt as a
small pybind11/`torch.utils.cpp_extension` package.

## Why this exists

The real `torchtext` package hasn't shipped a release since v0.18.0
(April 2024). Its compiled `_torchtext` extension is ABI-pinned to that era's
libtorch, so it can't be built or loaded against modern torch (e.g. 2.8.0+).
Its GitHub repo was archived (read-only) in September 2025.

Only `Vocab`, `build_vocab_from_iterator`/`vocab`, and `VocabTransform` were
needed here, so rather than resurrecting torchtext's full CMake build (which
also pulls in sentencepiece, re2, double-conversion, utf8proc as git
submodules), this extracts just those pieces into a self-contained extension
that builds cleanly against whatever torch is currently installed.

## What's included

- **`Vocab`** (`torchtext.vocab.Vocab`) — full API: `__len__`, `__contains__`,
  `__getitem__`, `__call__`/`forward` (token list → index list),
  `set_default_index`, `get_default_index`, `insert_token`, `append_token`,
  `lookup_token(s)`, `lookup_indices`, `get_stoi`, `get_itos`. Pickling works.
- **`vocab()`** / **`build_vocab_from_iterator()`** (`torchtext.vocab`) — same
  behavior as upstream: frequency sort (descending, then lexicographic tie
  break), `min_freq` filtering, `specials`/`special_first`, `max_tokens`.
- **`VocabTransform`** (`torchtext.transforms`) — wraps a `Vocab` as an
  `nn.Module`, accepts `List[str]` or `List[List[str]]`.
- Two lower-level, multithreaded C++ builders (mirroring torchtext's
  `vocab_factory.cpp`), reachable via `torchtext.vocab._torchtext`:
  - `_build_vocab_from_text_file_using_python_tokenizer(path, min_freq, tokenizer_fn)`
  - `_load_vocab_from_file(path, min_freq, num_cpus)` — assumes a
    whitespace-pre-tokenized file, parses it in parallel via `at::launch`.

## What's deliberately NOT included

- **`torchtext._torchtext.Vocab`** lives at `torchtext.vocab._torchtext`
  here instead of top-level `torchtext._torchtext` (upstream's location).
  Only matters if some other code imports that path directly.
- **`Vocab`'s module path differs**: upstream splits `vocab.py` (the class)
  from `vocab_factory.py` (the functions); this merges them into one
  `torchtext/vocab/vocab_factory.py`. Consequence: **a `Vocab` pickled by
  real torchtext 0.18 will NOT unpickle here, and vice versa** — pickle
  records the exact class module path.
- **No TorchScript export.** Upstream's `Vocab.__prepare_scriptable__` and
  the custom-class registration bootstrap (`torchtext._extension`) aren't
  ported, so `torch.jit.script(vocab_or_transform)` won't work.
- **No `Vectors`/`GloVe`/`FastText`/`CharNGram`** or any other torchtext
  submodule (`data`, `datasets`, `nn`, `models`, tokenizers, etc.) — only the
  vocab surface described above exists.
- **`_build_vocab_from_text_file`** (the variant that takes a
  `torch.jit.script.Module` as its tokenizer) is bound but raises
  `RuntimeError` on call. It depends on `torch::jit::as_module()`, an
  unstable/internal JIT-Python header, which SIGABRTs when compiled
  out-of-tree against a torch pip wheel (a pybind11 cross-module
  type-identity mismatch). Use
  `_build_vocab_from_text_file_using_python_tokenizer` instead.

## Naming caution

This package installs as `torchtext`, occupying the same import name as the
real PyPI package. Two consequences:

- If a directory literally named `torchtext/` exists on your `sys.path`
  (e.g. you `cd` into a torchtext source checkout), Python's cwd-first
  import resolution will shadow this installed package.
- Installing the real `torchtext` into the same environment will conflict
  outright.

## Building

Requires a C++17 compiler and whatever `torch` you want to build against
(tested against torch 2.10.0 / Python 3.13 on macOS arm64 — no CMake, no
submodules, just `torch.utils.cpp_extension`):

```bash
python setup.py bdist_wheel
pip install dist/torchtext-*.whl
```

The compiled `.so` is ABI-tied to the torch version/Python version/platform
it was built against — rebuild whenever any of those change.

**Import order matters**: `torch` must be imported before `torchtext.vocab`
loads its compiled extension, or the extension can't resolve
`libc10`/`libtorch` at load time. `torchtext/vocab/vocab_factory.py` already
does this for you internally, so this only matters if you're importing
`torchtext.vocab._torchtext` directly.

## Usage

```python
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.transforms import VocabTransform

def yield_tokens(lines):
    for line in lines:
        yield line.split()

v = build_vocab_from_iterator(yield_tokens(my_lines), specials=["<unk>"])
v.set_default_index(v["<unk>"])

v(["some", "tokens"])                 # -> List[int], via __call__/forward
v.lookup_indices(["some", "tokens"])  # same thing, explicit

vt = VocabTransform(v)
vt([["batch", "of"], ["token", "lists"]])  # -> List[List[int]]
```
