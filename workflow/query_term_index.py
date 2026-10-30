"""Load and scan a serialized token-level Aho-Corasick query-term index."""
import gzip
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


TOKEN_RE = re.compile(r"[^\W_]+|%", re.UNICODE)
FORMAT = "under25-query-term-ac-v1"
TYPE_PRIORITY = {"series_code": 0, "metric": 1, "financial_term": 2}


def token_key(value):
    return tuple(match.group(0).casefold() for match in TOKEN_RE.finditer(
        unicodedata.normalize("NFKC", value or "")))


@dataclass(frozen=True)
class QueryTermIndex:
    goto: list
    fail: list
    outputs: list
    postings: dict
    metadata: dict

    def scan(self, text):
        tokens=[(unicodedata.normalize("NFKC", match.group(0)).casefold(),match.start(),match.end())
                for match in TOKEN_RE.finditer(text)]
        raw=[];state=0
        for index,(token,_start,_end) in enumerate(tokens):
            while state and token not in self.goto[state]:state=self.fail[state]
            state=self.goto[state].get(token,0)
            for output_id in self.outputs[state]:
                posting=self.postings[str(output_id)]
                first=index-posting["length"]+1
                raw.append((tokens[first][1],tokens[index][2],posting))
        raw.sort(key=lambda row:(row[0],-(row[1]-row[0]),TYPE_PRIORITY.get(row[2]["type"],99)))
        kept=[];last_end=-1
        for start,end,posting in raw:
            if start<last_end:continue
            kept.append({"surface":text[start:end],"canonical":posting["canonical"],
                         "type":posting["type"],"types":posting.get("types",[posting["type"]]),
                         "aliases":posting.get("aliases",[]),"sources":posting.get("sources",[]),
                         "start":start,"end":end})
            last_end=end
        return kept


@lru_cache(maxsize=2)
def load_index(path):
    path=Path(path)
    try:
        data=json.loads(gzip.decompress(path.read_bytes()))
    except Exception as exc:
        raise ValueError("query_term_index_invalid: "+str(exc)) from exc
    if data.get("format")!=FORMAT:
        raise ValueError("query_term_index_format")
    return QueryTermIndex(data["goto"],data["fail"],data["outputs"],data["postings"],{
        key:value for key,value in data.items() if key not in {"goto","fail","outputs","postings"}})
