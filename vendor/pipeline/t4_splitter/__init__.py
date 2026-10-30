"""Offline, source-preserving O4000 splitter.

Only split_document imports the heavy draft splitter. configuration is standard
library only; count_tokens imports tiktoken and its bundled, verified vocabulary.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import threading

ROOT = Path(__file__).resolve().parent
ALGORITHM_VERSION = "optimized-strict-variable-v2"
PARAMETERS = {
    "draft_max_chunk_length": 4000,
    "draft_chunk_size": 4000,
    "draft_chunk_overlap": 200,
    "draft_content_types": ["text"],
    "is_sep_table_figure": True,
    "append_header_path": True,
    "only_allow_same_header_path": True,
    "debug": False,
    "optimize_min_core_tokens": 1800,
    "optimize_overlap_tokens": 200,
    "optimize_anaphora_backtrack_tokens": 500,
    "optimize_small_tail_tokens": 800,
    "optimize_merge_ceiling_tokens": 6000,
    "cap_limit": 4000,
    "cap_internal_budget": 3800,
    "source_mapping": "exact_unique_or_error",
    "blank_text": "error",
}
_encoding = None
_lock = threading.RLock()


class SplitterError(ValueError):
    """Input, dependency or source-mapping failure; never a silent fallback."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_resources() -> dict:
    expected = json.loads((ROOT / "RESOURCE_SHA256.json").read_text())
    for name, digest in expected.items():
        path = ROOT / name
        if not path.is_file() or _sha(path.read_bytes()) != digest:
            raise SplitterError(f"Bundled offline resource missing or changed: {name}")
    return expected


def _parameters(chunk_size: int) -> dict:
    if type(chunk_size) is not int or not 1000 <= chunk_size <= 8000:
        raise SplitterError('chunk_size must be an integer from 1000 to 8000')
    scale=chunk_size/4000
    return {**PARAMETERS,
        'draft_max_chunk_length':chunk_size,'draft_chunk_size':chunk_size,
        'optimize_min_core_tokens':round(1800*scale),
        'optimize_small_tail_tokens':round(800*scale),
        'optimize_merge_ceiling_tokens':round(6000*scale),
        'cap_limit':chunk_size,'cap_internal_budget':chunk_size-200}


def configuration(chunk_size: int = 4000) -> dict:
    """Return portable cache identity; no LlamaIndex/NLTK import or network."""
    resources = _verified_resources()
    code = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(ROOT))
        if rel not in resources:
            code[rel] = _sha(path.read_bytes())
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "tokenizer": {"name": "cl100k_base", "package": "tiktoken",
                      "version": "0.14.0", "allowed_special": "all",
                      "vocabulary_sha256": resources["resources/cl100k_base.tiktoken"]},
        "parameters": _parameters(chunk_size),
        "dependencies": json.loads((ROOT / "DEPENDENCIES.json").read_text()),
        "code_sha256": code,
        "resource_sha256": resources,
    }


def _require_versions(names: dict) -> None:
    for name, expected in names.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SplitterError(f"Missing {name}=={expected}; install requirements-index.txt") from exc
        if actual != expected:
            raise SplitterError(f"Expected {name}=={expected}, found {actual}; use locked index environment")


def _tokenize(text: str) -> list[int]:
    global _encoding
    if not isinstance(text, str):
        raise SplitterError("Tokenization requires a string")
    with _lock:
        if _encoding is None:
            _verified_resources()
            _require_versions({"tiktoken": "0.14.0"})
            import tiktoken
            # cl100k_base parameters from tiktoken 0.14.0 (MIT; licenses/tiktoken.txt).
            # Load the local ranks directly: get_encoding can otherwise fetch a URL.
            ranks = {base64.b64decode(token): int(rank) for token, rank in
                     (line.split() for line in (ROOT / "resources/cl100k_base.tiktoken").read_bytes().splitlines())}
            _encoding = tiktoken.Encoding(
                name="cl100k_base",
                pat_str=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s",
                mergeable_ranks=ranks,
                special_tokens={"<|endoftext|>": 100257, "<|fim_prefix|>": 100258,
                                "<|fim_middle|>": 100259, "<|fim_suffix|>": 100260,
                                "<|endofprompt|>": 100276},
            )
        return _encoding.encode(text, allowed_special="all")


def count_tokens(text: str) -> int:
    return len(_tokenize(text))


def _locate_draft(text: str, value: str, *, doc_id: str, seq: int) -> tuple[int, int]:
    start = text.find(value) if value else -1
    if start < 0:
        raise SplitterError(f"{doc_id}: draft {seq} cannot be mapped exactly to source; transformed/empty text")
    if text.find(value, start + 1) >= 0:
        raise SplitterError(f"{doc_id}: draft {seq} has ambiguous repeated text; refusing to guess source offset")
    return start, start + len(value)


def _drafts(doc_id: str, text: str, chunk_size: int) -> list[dict]:
    dependencies = json.loads((ROOT / "DEPENDENCIES.json").read_text())
    _require_versions(dependencies["direct"])
    _verified_resources()
    # Configure offline data before importing anything that may initialize NLTK.
    os.environ["NLTK_DATA"] = str(ROOT / "resources/nltk_data")
    import nltk
    nltk.data.path[:] = [os.environ["NLTK_DATA"]]
    from llama_index.core import Document
    from llama_index.core.utils import set_global_tokenizer, globals_helper
    from nltk.tokenize import PunktSentenceTokenizer
    set_global_tokenizer(_tokenize)
    # Same objects/data as LlamaIndex's helper, without its download fallback.
    globals_helper._stopwords = (ROOT / "resources/nltk_data/corpora/stopwords/english").read_text().splitlines()
    globals_helper._punkt_tokenizer = PunktSentenceTokenizer()
    from ._vendor.dc_parse.chunking.md_chunking import MdChunker
    from ._vendor.dc_parse.chunking.element_chunking import ElementChunker
    from ._vendor.dc_parse.chunking.sentence_chunking import SentenceChunker
    from ._vendor.dc_parse import utils
    utils.llama_tokenizer = _tokenize
    chunker = SentenceChunker(
        sub_parser=ElementChunker(sub_parser=MdChunker(), max_chunk_length=chunk_size,
                                  is_sep_table_figure=True, append_header_path=True,
                                  only_allow_same_header_path=True, debug=False),
        chunk_size=chunk_size, chunk_overlap=200, content_types=["text"], debug=False)
    nodes = chunker.cut_docs([Document(text=text, id_=doc_id)])
    if not nodes:
        raise SplitterError(f"{doc_id}: C4000 produced no draft chunks")
    drafts = []
    for seq, node in enumerate(nodes, 1):
        start, end = _locate_draft(text, node.text, doc_id=doc_id, seq=seq)
        if drafts and (start < drafts[-1]["start"] or end <= drafts[-1]["end"]):
            raise SplitterError(f"{doc_id}: draft {seq} is not in increasing source order")
        drafts.append({"seq": seq, "start": start, "end": end, "text": node.text})
    return drafts


def _pipe_regions(text: str) -> list[dict]:
    # The same diagnostic regions as the frozen final O4000 experiment.
    offset = 0
    start = None
    count = 0
    out = []
    for line in text.splitlines(keepends=True) + [""]:
        if line.count("|") >= 2:
            if start is None:
                start = offset
            count += 1
        else:
            if count >= 2:
                a, b = start, offset
                while a < b and text[a].isspace():
                    a += 1
                while b > a and text[b - 1].isspace():
                    b -= 1
                out.append({"start": a, "end": b, "kind": "原有逐行竖线表格"})
            start, count = None, 0
        offset += len(line)
    return out


def split_document(doc_id: str, text: str, chunk_size: int = 4000) -> list[dict]:
    """Dynamic C4000 → optimize → strict cap, preserving original characters."""
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise SplitterError("doc_id must be a nonempty string")
    if not isinstance(text, str) or not text.strip():
        raise SplitterError(f"{doc_id}: text must be a nonblank string")
    with _lock:
        # Imported only here, so cache reading stays lightweight.
        from .protected_split import optimize
        from .strict_cap import cap
        params=_parameters(chunk_size)
        drafts = _drafts(doc_id, text, chunk_size)
        protected = optimize(text, drafts, _tokenize,
            min_core=params['optimize_min_core_tokens'],small_tail=params['optimize_small_tail_tokens'],
            merge_ceiling=params['optimize_merge_ceiling_tokens'],label='O'+str(chunk_size))
        chunks = cap(text, protected["chunks"], _tokenize, limit=chunk_size)
        tables = protected["tables"]
        for region in tables:
            while region["start"] < region["end"] and text[region["start"]].isspace():
                region["start"] += 1
            while region["end"] > region["start"] and text[region["end"] - 1].isspace():
                region["end"] -= 1
        regions = tables + _pipe_regions(text)
        sections = protected["sections"]
        covered_until = 0
        source_hash = _sha(text.encode("utf-8"))
        for seq, chunk in enumerate(chunks, 1):
            a, b = chunk["start"], chunk["end"]
            if not (0 <= a < b <= len(text)) or a > covered_until:
                raise SplitterError(f"{doc_id}: invalid interval or source gap at chunk {seq}")
            if chunk["text"] != text[a:b] or chunk["tokens"] != count_tokens(text[a:b]) or chunk["tokens"] > chunk_size:
                raise SplitterError(f"{doc_id}: source/token invariant failed at chunk {seq}")
            covered_until = max(covered_until, b)
            active = [h for h in sections if h["start"] <= chunk.get("core_start", a)]
            chunk["section"] = active[-1]["label"] if active else "未恢复章节"
            chunk["sections_crossed"] = [h["label"] for h in sections if a < h["start"] < b]
            chunk["split_regions"] = [{"start": r["start"], "end": r["end"], "kind": r["kind"]}
                                      for r in regions if a < r["end"] and b > r["start"]
                                      and not (a <= r["start"] and b >= r["end"])]
            chunk["table_regions"] = sum(a <= r["start"] and r["end"] <= b for r in tables)
            chunk["notes"] = "保护区域被拆分，需按范围补读相邻块" if chunk["split_regions"] else ""
            chunk.update(doc_id=doc_id, source_sha256=source_hash,
                         algorithm_version=ALGORITHM_VERSION,
                         evidence_id=f"{doc_id}::{a}::{b}")
        if not chunks or covered_until != len(text):
            raise SplitterError(f"{doc_id}: final chunks do not cover the complete source")
        return chunks
