# _*_coding:utf-8_*_

from pydantic import BaseModel, Field
from simhash import Simhash
import jieba
import re
from pathlib import Path
from typing import List, Tuple, Iterable

from llama_index.core.utils import get_tokenizer
from typing import Callable

llama_tokenizer = get_tokenizer()

def len_tok(text: str) -> int:
    return len(llama_tokenizer(text))

class Chunk(BaseModel):
    context_gen: str
    text: str
    simhash: int
    embedding: str = Field(default="")
    embeding_dim: int = Field(default=0)
    id_code: str = Field(default="")
    item_id: str = Field(default="")
    content_type: str = Field(default="text")
    tables: list = Field(default=[])
    header_path: str = Field(default="")


_STOPWORD_DIR = Path(__file__).parent
STOPWORDS = [
    w.strip()
    for w in (
        _STOPWORD_DIR.joinpath("stop_words.ch.txt").read_text(encoding="utf-8").splitlines()
        + _STOPWORD_DIR.joinpath("stop_words.en.txt").read_text(encoding="utf-8").splitlines()
    )
    if w.strip()
]

HEADER_LINE_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_header_title(title: str) -> str:
    title = collapse_spaces(title)
    title = re.sub(r"\s+#+\s*$", "", title)
    return collapse_spaces(title)


def escape_header_path_slash(part: str) -> str:
    """Encode slashes inside a title so they are not parsed as path separators."""
    return part.replace("/", "%2F")


def split_header_path(path: str) -> List[str]:
    path = collapse_spaces(path)
    if not path or path == "/":
        return []
    return [collapse_spaces(part) for part in path.split("/") if collapse_spaces(part)]


def build_header_path(parts: Iterable[str]) -> str:
    normalized_parts = [collapse_spaces(part) for part in parts if collapse_spaces(part)]
    if not normalized_parts:
        return "/"
    return "/" + "/".join(normalized_parts) + "/"


def combine_sibling_header_paths(left_path: str, right_path: str) -> str | None:
    left_parts = split_header_path(left_path)
    right_parts = split_header_path(right_path)
    if (
        left_parts
        and len(left_parts) == len(right_parts)
        and left_parts != right_parts
        and left_parts[:-1] == right_parts[:-1]
    ):
        return build_header_path([*right_parts[:-1], f"{left_parts[-1]}&{right_parts[-1]}"])
    return None


def combine_sibling_header_path_sequence(paths: Iterable[str]) -> str | None:
    normalized_paths = [split_header_path(path) for path in paths if split_header_path(path)]
    if len(normalized_paths) < 2:
        return None

    base_parts = normalized_paths[-1]
    if any(
        len(parts) != len(base_parts)
        or parts[:-1] != base_parts[:-1]
        for parts in normalized_paths
    ):
        return None

    leaves = list(dict.fromkeys(parts[-1] for parts in normalized_paths))
    if len(leaves) < 2:
        return None
    return build_header_path([*base_parts[:-1], "&".join(leaves)])


def normalize_single_header_path(path: str) -> str:
    return build_header_path(split_header_path(path))


def is_root_header_path(path: str) -> bool:
    return normalize_single_header_path(path) == "/"


def header_path_depth(path: str) -> int:
    return len(split_header_path(path))


def iter_header_paths(raw_paths: Iterable[str]) -> List[str]:
    normalized_paths: List[str] = []
    seen: set[str] = set()

    for raw_path in raw_paths:
        if raw_path is None:
            continue
        for part in str(raw_path).split(";"):
            normalized = normalize_single_header_path(part)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_paths.append(normalized)

    return normalized_paths


def pick_primary_header_path(raw_paths: Iterable[str]) -> str:
    candidates = iter_header_paths(raw_paths)
    if not candidates:
        return "/"

    best_index = 0
    best_score = (-1, -1, -1, -1)
    for index, path in enumerate(candidates):
        score = (
            0 if is_root_header_path(path) else 1,
            header_path_depth(path),
            len(path),
            index,
        )
        if score > best_score:
            best_index = index
            best_score = score

    return candidates[best_index]


def extract_header_path_candidates(
    text: str,
    base_path: str = "/",
    escape_header_path_slashes: bool = False,
) -> List[str]:
    base_parts = split_header_path(base_path)
    current_parts = list(base_parts)
    candidates: List[str] = []

    for match in HEADER_LINE_RE.finditer(text or ""):
        level = len(match.group(1))
        title = normalize_header_title(match.group(2))
        if not title:
            continue
        if escape_header_path_slashes:
            title = escape_header_path_slash(title)

        keep_depth = level - 1
        if keep_depth < len(current_parts):
            current_parts = current_parts[:keep_depth]
        current_parts = current_parts + [title]
        candidates.append(build_header_path(current_parts))

    return candidates


def derive_canonical_header_path(
    text: str,
    base_path: str = "/",
    escape_header_path_slashes: bool = False,
) -> str:
    return pick_primary_header_path(
        [
            base_path,
            *extract_header_path_candidates(
                text,
                base_path,
                escape_header_path_slashes=escape_header_path_slashes,
            ),
        ]
    )

def gen_ngram(word_list, ngram='+2', ch_concat=True):
    ## 只拼接相邻单字
    # wlens = [len(x) for x in word_list]
    lens = [len(x)==1 for x in word_list]

    if ngram[0] == '+':
        ngrams = [k for k in range(int(ngram[1:]), len(word_list)+1, 1)]
    elif ngram[0] == '-':
        ngrams = [k for k in range(int(ngram[1:]), 1, -1)]
    else:
        ngrams = [int(ngram)]
    ret = []
    for ngram in ngrams:
        temp_sub = zip(*[word_list[i:] for i in range(0, ngram)])
        temp_len = zip(*[lens[i:] for i in range(0, ngram)])
        for x, l in zip(temp_sub, temp_len):
            gram = ''.join(x)
            if len(gram) == len(gram.replace(' ', '')):
                # 只拼接单字符组成的ngram
                if ch_concat:
                    if sum(l) == len(l):
                        ret.append(gram)
                else:
                    ret.append(gram)

    return list(set(ret))

def run_sim_hash(text):
    ## 需要做分词、去停用词的操作
    ## md 中的表格 存在 高频特殊符号，会影响simhash 结果
    words = jieba.lcut(text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    ## 前100， 最后100 算bigrams

    ngrams = gen_ngram(words[:100]) + gen_ngram(words[-100:]) + words  # 2-gram + unigram
    return Simhash(ngrams).value


seperator_chinese = "。？！；"

def split_chinese_sentences(doc):
    line_break = re.compile('[\r\n]')
    # delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        pattern = '(?<=[，。？！；])'
        # 分割文本，保留符号
        s_sentences = re.split(pattern, line)

        for sent in s_sentences:
        # for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)

        sentences[-1] += '\n' # last keep \n
    return sentences


def get_chinese_ratio(text):
    ch_blocks = re.findall('[\u4e00-\u9fff]+', text)
    total = len(text)
    ch_num = sum([len(ch) for ch in ch_blocks])
    return ch_num / total

### all metadata fields

TYPE_META_TABLE = "tables"
TYPE_META_CONTENT_TYPE = "content_type"
TYPE_META_HEADER_PATH = "header_path"

METADATA_FIELDS = [
    TYPE_META_TABLE,
    TYPE_META_CONTENT_TYPE,
    TYPE_META_HEADER_PATH
]


def replace_special(md_text):
    """ 替换md 中的特殊字符"""
    pattern = r'^#{1,6}\s+(.*)$'
    # 替换为加粗格式：**标题文本**
    replaced = re.sub(pattern, r'**\1**', md_text, flags=re.MULTILINE)
    return replaced


def split_pattern_and_rep_title(text: str, pattern: str, tp: str) -> List[str]:
    ret_list = []
    offset = 0
    while offset < len(text):
        # match = re.search(pattern, text[offset:], re.MULTILINE|re.DOTALL)
        match = re.search(pattern, text[offset:], re.DOTALL)
        if match:
            start, end = match.start() + offset, match.end() + offset
            content_match = text[start: end]

            if offset < start and text[offset: start].strip():
                ret_list.append({'content': text[offset: start], 'type': 'text'})

            content_match = replace_special(content_match)

            ret_list.append({'content': content_match, 'type': tp})
            offset = end

        else:
            ret_list.append({'content': text[offset: len(text)], 'type': 'text'})
            break

    return ret_list

def is_md_title(md_text: str) -> bool:
    pattern = r'^(#{1,6}\s+.*)$'
    return re.match(pattern, md_text, re.IGNORECASE) is not None
