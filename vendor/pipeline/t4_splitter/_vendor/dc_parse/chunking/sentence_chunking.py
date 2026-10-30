# _*_coding:utf-8_*_

from .md_chunking import MdChunker, Chunker
from typing import Any, List
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from pydantic import BaseModel, Field
from ..utils import Chunk, split_chinese_sentences, seperator_chinese, get_chinese_ratio
from ..utils import METADATA_FIELDS, TYPE_META_CONTENT_TYPE
from llama_index.core.node_parser import SentenceSplitter
from typing import Callable
import json
import logging
import re


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """
    Split text with separator and keep the separator at the end of each split.

    Args:
        text (str): The text to split.
        separator (str): The separator to split on.

    Returns:
        List[str]: List of text segments with separators preserved at the end of each split.

    """
    # result = re.findall(f'(.+?'+separator+')', text)
    # return result

    ret_list = []
    offset = 0
    pattern = f'(.+?['+separator+'])'
    while offset < len(text):
        match = re.search(pattern, text[offset:], re.DOTALL)
        if match:
            start, end = match.start() + offset, match.end() + offset
            content_match = text[start: end]

            if offset < start and text[offset: start].strip():
                ret_list.append(text[offset: start])

            ret_list.append(content_match)
            offset = end

        else:
            ret_list.append(text[offset: len(text)])
            break

    return ret_list


def split_by_sep(sep: str, keep_sep: bool = True) -> Callable[[str], List[str]]:
    """
    Create a function that splits text by a separator.

    Args:
        sep (str): The separator to split on.
        keep_sep (bool, optional): Whether to keep the separator in the output. Defaults to True.

    Returns:
        Callable[[str], List[str]]: A function that takes a string and returns a list of split strings.

    """
    if keep_sep:
        return lambda text: split_text_keep_separator(text, sep)
    else:
        return lambda text: text.split(sep)


class SentenceChunker(Chunker):
    chunk_size: int = Field(default=1024)  # 根据 tokens 来算的
    chunk_overlap: float = Field(default=20)
    chinese_ratio_thres: float = Field(default=0.30)

    splitter_ch: SentenceSplitter = Field(default=None)
    splitter_en: SentenceSplitter = Field(default=None)
    content_types: List[str] = Field(default=[], description="只对对指定类型的元素进行处理，其他元素跳过处理（比如figure/table）")
    debug: bool = Field(default=False, description="debug")


    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.splitter_ch = SentenceSplitter(chunk_size=self.chunk_size,
                                         chunk_overlap=self.chunk_overlap)
        # separator = seperator_chinese)

        self.splitter_ch._split_fns[0] = split_by_sep(seperator_chinese)


        self.splitter_en = SentenceSplitter(chunk_size=self.chunk_size,
                                            chunk_overlap=self.chunk_overlap)

    def get_id_code(self, node, offset:int):
        return str(int(offset))

    def gen_from_docs(self, documents: List[Document]) -> List[BaseNode]:
        ret_nodes = []
        for doc in documents:
            if get_chinese_ratio(doc.text) > self.chinese_ratio_thres:
                nodes = self.splitter_ch.get_nodes_from_documents([doc])
            else:
                nodes = self.splitter_en.get_nodes_from_documents([doc])

            for idx, node in enumerate(nodes):
                node.metadata['doc_id'] = doc.id_

                ## item_id: ${doc_id}_${offset}
                node.metadata['id_code'] = str(idx)
                ret_nodes.append(node)

            ret_nodes += nodes
        return nodes

    def gen_from_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        ret_nodes = []
        global_id_code = 0

        if self.debug:
            with open('__ele_subparer_results.json', 'w') as f:
                arr = [{"idx": idx, "text": node.get_content(), "meta": node.metadata}
                       for idx, node in enumerate(nodes)]
                json.dump({'arr': arr}, f, ensure_ascii=False)

        for idx, node in enumerate(nodes):
            logging.debug(f"idx: {idx}, node len: {len(node.get_content())}")

            content_type = node.metadata.get(TYPE_META_CONTENT_TYPE, "")
            # skip
            if self.content_types and content_type not in self.content_types:
                node.metadata['id_code'] = str(global_id_code)
                ret_nodes.append(node)
                global_id_code +=1

                logging.debug('\t', len(node.get_content()))
                continue

            text = node.get_content()
            if get_chinese_ratio(text) > self.chinese_ratio_thres:
                sub_nodes = self.splitter_ch.get_nodes_from_documents([Document(text=text)])
            else:
                sub_nodes = self.splitter_en.get_nodes_from_documents([Document(text=text)])


            for sub_node in sub_nodes:
                ## copy field from parent
                for k in METADATA_FIELDS:
                    if node.metadata.get(k):
                        sub_node.metadata[k] = node.metadata.get(k)

                sub_node.metadata['id_code'] = str(global_id_code)
                ret_nodes.append(sub_node)

                global_id_code+=1

            # print(f"\t, {sum([len(s.get_content()) for s in sub_nodes]), [len(s.get_content()) for s in sub_nodes]}")

        return ret_nodes





