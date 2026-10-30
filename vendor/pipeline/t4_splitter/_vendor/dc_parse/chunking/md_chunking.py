# _*_coding:utf-8_*_
from xml.dom.minidom import Document

from pydantic import BaseModel, Field
from typing import List
from llama_index.core.schema import BaseNode, TextNode
from pydantic import BaseModel
from llama_index.core.node_parser import MarkdownNodeParser, MarkdownElementNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser.interface import NodeParser
from typing import ForwardRef
from ..utils import Chunk, run_sim_hash, len_tok, derive_canonical_header_path, pick_primary_header_path
from  llama_index.core.schema import Document
from typing import Dict
from ..utils import METADATA_FIELDS, TYPE_META_CONTENT_TYPE
from ..utils import replace_special, split_pattern_and_rep_title
import re
import numpy as np


class Chunker(BaseModel):
    sub_parser: ForwardRef('Chunker') = Field(default=None)
    chunk_header_path_mode: str = Field(default="primary")
    @property
    def chunker_code(self):
        return ""

    def cut_file(self, file_name:str) -> List[BaseNode]:
        """从文件切分"""
        if not self.sub_parser:
            documents = SimpleDirectoryReader(input_files=[file_name]).load_data()
            nodes = self.gen_from_docs(documents)
        else:
            nodes = self.sub_parser.cut_file(file_name)
            nodes = self.gen_from_nodes(nodes)

        return nodes


    def cut_docs(self, documents: List[Document]) -> List[BaseNode]:
        """从docs 切分"""
        if not self.sub_parser:
            nodes = self.gen_from_docs(documents)
        else:
            nodes = self.sub_parser.cut_docs(documents)
            nodes = self.gen_from_nodes(nodes)

        return nodes


    def gen_from_docs(self, documents: List[Document]) -> List[BaseNode]:
        return []

    def gen_from_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        return nodes

    def _finalize_chunk_header_path(self, header_path: str | None) -> str | None:
        if header_path is None:
            return None
        if self.chunk_header_path_mode == "primary":
            return pick_primary_header_path([header_path])
        if self.chunk_header_path_mode == "keep":
            return header_path
        raise ValueError(f"Unknown chunk_header_path_mode: {self.chunk_header_path_mode}")

    def _collect_chunk_metadata(self, node: BaseNode) -> Dict:
        metadata = dict([(k, node.metadata.get(k)) for k in METADATA_FIELDS if node.metadata.get(k)])
        header_path = self._finalize_chunk_header_path(node.metadata.get('header_path'))
        if header_path:
            metadata['header_path'] = header_path
        else:
            metadata.pop('header_path', None)
        return metadata

    def convert_node_to_chunk(self, node: BaseNode, doc_id: str) -> Chunk:
        return Chunk(context_gen=node.metadata.get('context_gen', ""),
                     text=node.text,
                     simhash=run_sim_hash(node.text),
                     item_id=self._gen_item_id(node, doc_id),
                     **self._collect_chunk_metadata(node)
                     )

    def _gen_item_id(self, node, doc_id:str):
        id_code = node.metadata.get('id_code', "")
        if id_code:
            item_id = str(doc_id) + '_' + id_code
        else:
            item_id = str(doc_id)
        return item_id


class MdChunker(Chunker):
    """
    读入md文件及元数据文件，输出chunks结果
    """
    parser: MarkdownNodeParser = Field(default=MarkdownNodeParser())
    # parser: MarkdownElementNodeParser = Field(default=MarkdownElementNodeParser())
    #pattern_rep: str = Field(default=r'!\[.*?\]\(.*?(?:"\)|\))')  # r'!\[.*?\]\(.*?\"\)'
    pattern_rep: str = Field(default=r'!\[.*?\]\(.*?(?:"\))')  # r'!\[.*?\]\(.*?\"\)'
    cutoff_figure_desc: int = Field(default=1024, description="最大字符个数")

    header_path_mode: str = Field(default="canonical")
    escape_header_path_slashes: bool = Field(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _cut_figure_desc(self, obj):

        content = obj['content']
        idx = content.find('](')
        start_desc = content.find(',', idx+2)

        if content[idx:start_desc].find('data:image/png;base64') != -1:
            ## 处理base64的图, 非法 figure 格式
            return None

        split_pattern = "([\"]*)([^\"]+)([\"]*\))"
        ## description

        match= re.match(split_pattern, content[start_desc + 1:].strip())

        if match:
            groups = list(match.groups())
            if len(groups) == 3:
                idx_desc = np.argmax([len(g) for g in groups]).tolist()

                groups[idx_desc] = groups[idx_desc][:self.cutoff_figure_desc]
                content= content[:start_desc+1] + " " + ''.join(groups)
                # print(f"content: {content}")

                obj['content'] = content

        return obj

    def _process_figure(self, md_text):
        """ 替换表描述中的特殊字符，对表描述进行截断 """
        ret_list = split_pattern_and_rep_title(md_text, pattern = self.pattern_rep, tp='figure')
        ## cutoff
        ret_list = [self._cut_figure_desc(r) if r['type']=='figure' else r for r in ret_list]
        contents = [r['content'] for r in ret_list if r]
        return ''.join(contents)

    def _build_header_path(self, text: str, base_header_path: str | None) -> str | None:
        if self.header_path_mode == "canonical":
            return derive_canonical_header_path(
                text,
                base_header_path or '/',
                escape_header_path_slashes=self.escape_header_path_slashes,
            )
        if self.header_path_mode == "legacy":
            return base_header_path
        raise ValueError(f"Unknown header_path_mode: {self.header_path_mode}")

    def gen_from_docs(self, documents: List[Document]) -> List[BaseNode]:
        ret_nodes: List[BaseNode] = []
        for doc in documents:
            doc.set_content(self._process_figure(doc.text))
            nodes = self.parser.get_nodes_from_documents([doc])
            for idx, node in enumerate(nodes):
                node.metadata['doc_id'] = doc.id_
                ## item_id: ${doc_id}_${offset}
                node.metadata['id_code'] = str(idx)
                node.metadata['header_path'] = self._build_header_path(
                    node.get_content(),
                    node.metadata.get('header_path', '/'),
                )
                ret_nodes.append(node)

        return ret_nodes

class MdElementChunker(Chunker):
    """
    读入md文件及元数据文件，输出chunks结果
    """
    parser: MarkdownElementNodeParser = Field(default=MarkdownElementNodeParser())
    header_path_mode: str = Field(default="canonical")
    escape_header_path_slashes: bool = Field(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_header_path(self, text: str, base_header_path: str | None) -> str | None:
        if self.header_path_mode == "canonical":
            return derive_canonical_header_path(
                text,
                base_header_path or '/',
                escape_header_path_slashes=self.escape_header_path_slashes,
            )
        if self.header_path_mode == "legacy":
            return base_header_path
        raise ValueError(f"Unknown header_path_mode: {self.header_path_mode}")

    def gen_from_docs(self, documents: List[Document]) -> List[BaseNode]:
        ret_nodes: List[BaseNode] = []
        for doc in documents:
            elements = self.parser.extract_elements(text=doc.get_content())
            for idx, element in enumerate(elements):
                node = TextNode(text=element.element)
                node.metadata['doc_id'] = doc.id_
                node.metadata['id_code'] = str(idx)
                node.metadata[TYPE_META_CONTENT_TYPE] = element.type
                node.metadata['header_path'] = self._build_header_path(
                    node.get_content(),
                    node.metadata.get('header_path', '/'),
                )
                ret_nodes.append(node)

        return ret_nodes



