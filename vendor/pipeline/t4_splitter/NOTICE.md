# O4000 source and offline data provenance

This package preserves the frozen parse-pipeline C4000 draft algorithm followed
by `protected_split.optimize` and `strict_cap.cap(4000)`. Source SHA256 values
are recorded in `SOURCE_SHA256.json`; offline resource SHA256 values are in
`RESOURCE_SHA256.json`. Both manifests and every package source/resource are
included in `configuration()` for cache identity.

The three chunkers, utility module and stopword lists were copied from the
user-provided `experiments/md-index/vendor/parse-pipeline/src/dc_parse`
snapshot. Only `from dc_parse.utils` imports were changed to package-relative
imports. No LICENSE or NOTICE was present in that snapshot or its supplied
original repository. No third-party license is invented here; confirm the
source's redistribution permission before publishing this package externally.

`protected_split.py` is unchanged. `strict_cap.py` changes only the sibling
import to a relative import. Both originate in this workspace's experiments.
Wrapper validation, metadata refresh and local resource initialization are new.

The bundled cl100k_base vocabulary is the existing experiment cache, SHA256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.
Its encoding configuration comes from tiktoken 0.14.0; tiktoken's MIT notice is
retained in `licenses/tiktoken.txt`. Loading is directly from local bytes and
does not invoke tiktoken's URL/cache loader.

NLTK resources are the supplied English stopwords and English punkt_tab files,
with their original dataset README files retained. They are bundled under
`resources/nltk_data`; no runtime path points to Downloads. Only English punkt
resources are included because the frozen chunker uses its own Chinese
separator and LlamaIndex's default English sentence tokenizer. NLTK and
LlamaIndex package licenses are retained in `licenses/` as attribution; the
packages themselves are installed separately via `requirements-index.txt`.

Build the index with Python 3.12 and the locked dependencies. Reading an index
only needs Python 3.10+ and tiktoken 0.14.0. The new wrapper rejects blank input,
transformed or ambiguously repeated draft text, source gaps and wrong token
lengths explicitly; it does not switch to another splitting algorithm.
