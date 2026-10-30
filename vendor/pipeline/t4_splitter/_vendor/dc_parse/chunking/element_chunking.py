# _*_coding:utf-8_*_
import logging

from .md_chunking import Chunker
from typing import Any, List
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from pydantic import BaseModel, Field
from ..utils import (
    replace_special,
    split_pattern_and_rep_title,
    is_md_title,
    pick_primary_header_path,
    build_header_path,
    combine_sibling_header_path_sequence,
    split_header_path,
)
import re
import json
from ..utils import METADATA_FIELDS, TYPE_META_TABLE, TYPE_META_CONTENT_TYPE, TYPE_META_HEADER_PATH
from ..utils import len_tok

import itertools
from llama_index.core.schema import TextNode

from markdown_it import MarkdownIt
md_parser = MarkdownIt("commonmark").enable('table')


class ElementChunker(Chunker):
    """
    根据元素类型进行切分
    1、表格、图片 单独成段
    2、文字合并，不超过长度限制
    """
    min_chunk_length: int = Field(default=50,description="低于min_chunk_length 不单独成段")
    min_chunk_length_after: int = Field(default=100,description="图表后内容，不超过该长度，允许合并到图里")
    max_chunk_length: int = Field(default=1024,description="基于md输出结果合并，合完尽量不超过字符数")
    is_sep_table_figure: bool = Field(default=False, description="表、图是否独立成段")
    append_header_path: bool = Field(default=False, description="chunk是否保存 md的header path")
    append_table: bool = Field(default=False, description="解析表结构")
    only_allow_same_header_path: bool = Field(default=False, description="合并需要header path相同")
    merged_header_path_mode: str = Field(default="primary", description="header_path merge policy: primary, multi, or last")
    allow_title_prefix_header_merge: bool = Field(default=True, description="allow title prefix merge across header path")
    merge_small_header_path_text: bool = Field(default=False, description="merge title-only or short heading text across header_path")
    small_header_path_text_max_tokens: int = Field(default=120, description="max tokens for short heading text cross-header merge")
    debug: bool = Field(default=False, description="debug")

    def get_id_code(self, node, offset:int):
        return str(int(offset))

    def gen_from_docs(self, documents: List[Document]) -> List[BaseNode]:
        raise NotImplementedError("必须包含subparser")

    def _get_content_from_md_node(self, node_arr):
        ## 将markdown 的header path拼上
        text_arr = []
        for n in node_arr:
            # if self.append_header_path and bool(re.search(r'[\u4e00-\u9fffa-zA-Z]',header_path)):
            #     content = '\n\n'.join(["Context: "+header_path, n.get_content()])
            # else:
            content = n.get_content()
            # print(f"\t\t\t content: {content}")
            text_arr.append(content)
        return '\n'.join(text_arr)

    def _new_node(self, node_arr: List[BaseNode], offset: int) -> BaseNode:
        # model_copy shares nested metadata, so derive this before writing the
        # default merged metadata onto the copied first node.
        title_prefix_path = self._get_title_prefix_table_header_path(node_arr)
        new_node: BaseNode = node_arr[0].model_copy()
        content = self._get_content_from_md_node(node_arr)

        new_node.metadata['id_code'] = self.get_id_code(node_arr[0], offset)
        node_types = [n.metadata.get(TYPE_META_CONTENT_TYPE,'') for n in node_arr]
        if 'table' in node_types:
            node_type = 'table'
        elif 'figure' in node_types:
            node_type = 'figure'
        else:
            node_type = 'text'

            # new_node.metadata[TYPE_META_CONTENT_TYPE] = node_type
        new_node.metadata[TYPE_META_CONTENT_TYPE] = node_type

        new_node.set_content(content)
        ## 需要合并的metadata
        for key in [TYPE_META_TABLE]:
            new_node.metadata[key] = list(itertools.chain(*[n.metadata.get(key)
                                                          for n in node_arr if key in n.metadata]))
        ## 需要去重合并的metadata
        for key in [TYPE_META_HEADER_PATH]:
            header_paths = [n.metadata.get(key) for n in node_arr if key in n.metadata]

            new_node.metadata[key] = self._merge_header_paths(header_paths)

        if title_prefix_path:
            new_node.metadata[TYPE_META_HEADER_PATH] = title_prefix_path

        return new_node

    @staticmethod
    def _get_title_prefix_table_header_path(node_arr: List[BaseNode]) -> str | None:
        """Combine sibling title paths when they are a pure prefix of a table."""
        if len(node_arr) < 2:
            return None

        table_node = node_arr[-1]
        if table_node.metadata.get(TYPE_META_CONTENT_TYPE, "text") != "table":
            return None

        title_nodes = node_arr[:-1]
        if any(
            node.metadata.get(TYPE_META_CONTENT_TYPE, "text") != "text"
            or not ElementChunker.is_standalone_title_text(node.get_content())
            for node in title_nodes
        ):
            return None

        title_paths = [split_header_path(node.metadata.get(TYPE_META_HEADER_PATH, "/")) for node in title_nodes]
        table_path = split_header_path(table_node.metadata.get(TYPE_META_HEADER_PATH, "/"))
        if not table_path or any(not path for path in title_paths):
            return None

        # MarkdownNodeParser can keep the last title and its following table
        # in the same table node. In that form the table path represents the
        # final title in the otherwise pure-title prefix.
        if not (
            ElementChunker.has_leading_md_title(table_node.get_content())
            or title_paths[-1] == table_path
        ):
            return None

        all_paths = [*title_paths, table_path]
        if any(
            len(path) != len(table_path)
            or path[:-1] != table_path[:-1]
            for path in all_paths
        ):
            return None

        return combine_sibling_header_path_sequence(
            build_header_path(path) for path in all_paths
        )

    def _merge_header_paths(self, header_paths: List[str]) -> str:
        normalized_paths = [path for path in header_paths if path]
        if self.merged_header_path_mode == "primary":
            return pick_primary_header_path(normalized_paths)
        if self.merged_header_path_mode == "multi":
            return '; '.join(dict.fromkeys(normalized_paths))
        if self.merged_header_path_mode == "last":
            return normalized_paths[-1] if normalized_paths else "/"
        raise ValueError(f"Unknown merged_header_path_mode: {self.merged_header_path_mode}")


    @staticmethod
    def extract_tables_from_tokens(tokens):
        tables = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == "table_open":
                # 定位表格子树（直到 table_close）
                table_tokens = []
                depth = 0
                j = i
                while j < len(tokens):
                    t = tokens[j]
                    if t.type == "table_open":
                        depth += 1
                    elif t.type == "table_close":
                        depth -= 1
                    table_tokens.append(t)
                    j += 1
                    if depth == 0:
                        break

                # 解析表格内容
                table_data = ElementChunker._parse_table_tokens(table_tokens)
                tables.append(table_data)
                i = j  # 跳过已处理的 tokens
            else:
                i += 1
        return tables

    @staticmethod
    def _parse_table_tokens(table_tokens):
        # 简化版：只取 <tr> 中的 <th>/<td> 内的 inline 文本
        rows = []
        current_row = None
        in_header = False

        for token in table_tokens:
            if token.type == "tr_open":
                current_row = []
            elif token.type == "tr_close" and current_row is not None:
                rows.append(current_row)
                current_row = None
            elif token.type in ("th_open", "td_open"):
                # 找到下一个 inline token 作为内容
                # （实际中可能需遍历 children，此处简化）
                pass
            elif token.type == "inline" and current_row is not None:
                # 注意：inline 可能包含多个子 token（如强调、链接）
                # 这里用 renderInline 简化获取纯文本
                from markdown_it import MarkdownIt
                content = MarkdownIt().renderInline(token.content).strip()
                current_row.append(content)

        # 第一行是表头，后续是数据
        if len(rows) >= 2:
            return {
                "header": rows[0],
                "rows": rows[1:]
            }
        return {"header": [], "rows": []}


    @staticmethod
    def split_by_figure_or_table(node: BaseNode, add_table_meta=False,
                                 figure_pattern=r'!\[.*?\]\(.*?\"\)',
                                 min_text_len=50,
                                 min_text_len_after=100, # 图内容 后的文字可以归并的长度
                                 ) -> List[BaseNode]:
        global md_parser
        text = node.text
        # figure_pattern = r'!\[.*?\]\(.*?\"\)'
        # # 匹配Markdown表格语法（至少包含一行 | 和 -）
        # table_pattern = r'\|.*\|\n\s*\|[-| ]+\|'

        res_list = split_pattern_and_rep_title(text, figure_pattern, 'figure')
        node_buffer = []
        new_nodes = []

        ## 允许前序title 与figure|table 合并
        """
        #### Fig. 2: Export growth and its sources by key markets\n![Fig. 2: Export growth and its sour](1894315827751157760_2.png, \"这是...\")
        """
        n_idx= 0
        while n_idx < len(res_list):
            obj = res_list[n_idx]
            if obj['type'] == 'text':
                try:
                    tokens = md_parser.parse(obj['content'])
                    tables = ElementChunker.extract_tables_from_tokens(tokens)
                    obj[TYPE_META_TABLE] = [t for t in tables if t]
                    if obj[TYPE_META_TABLE]:
                        obj['type'] = 'table'

                    text = obj['content']

                    ## 当前是title 或着 长度小于指定长度
                    if ElementChunker.can_rotate(text, [text], min_text_len=min_text_len):
                        node_buffer += [obj]
                        n_idx += 1
                        continue
                    else:
                        text = obj['content']
                        node_buffer = []
                        new_node = TextNode(text=text, metadata=node.metadata.copy())
                        if add_table_meta:
                            new_node.metadata[TYPE_META_TABLE] = obj[TYPE_META_TABLE]

                        new_node.metadata[TYPE_META_CONTENT_TYPE] = obj['type']
                        new_nodes.append(new_node)

                except Exception as e:
                    logging.error(e)
            else:
                ## 图内容
                obj[TYPE_META_TABLE] = []
                node_buffer += [obj]

                # 允许合并后续的文字
                if n_idx+1 < len(res_list) and ElementChunker.can_rotate_after(res_list[n_idx+1]['content'],
                                                                               min_text_len_after):
                    node_buffer += [res_list[n_idx+1]]
                    n_idx += 1

                text = ''.join([n['content'] for n in node_buffer])
                node_buffer = []
                new_node = TextNode(text=text, metadata=node.metadata.copy())
                if add_table_meta:
                    new_node.metadata[TYPE_META_TABLE] = obj[TYPE_META_TABLE]

                new_node.metadata[TYPE_META_CONTENT_TYPE] = obj['type']
                new_nodes.append(new_node)

            n_idx += 1

        if node_buffer:
            text = '\n'.join([n['content'] for n in node_buffer])
            new_node = TextNode(text=text, metadata=node.metadata.copy())
            if add_table_meta:
                new_node.metadata[TYPE_META_TABLE] = obj[TYPE_META_TABLE]

            new_node.metadata[TYPE_META_CONTENT_TYPE] = obj['type']
            new_nodes.append(new_node)

        return new_nodes

    @staticmethod
    def calc_title_ratio(history_arr: List[BaseNode]|List[str]):
        ## 最后节点是 title
        ## title 占比足够多

        pattern = r'^(#{1,6}\s+.*)$'
        all_contents = []
        all_titles = []

        for n in history_arr:
            if isinstance(n, TextNode):
                md_text = n.get_content().strip()
            else:
                md_text = n
            all_contents.append(md_text)

            matches = re.findall(pattern, md_text, flags=re.MULTILINE)
            titles = [title for title in matches]

            all_titles.extend(titles)

        # print("all_titles:", all_titles)
        # print("all_contents:", all_contents)
        #
        # print([re.sub('\s+', '', title) for title in all_titles])
        # print([re.sub('\s+', '', content) for content in all_contents])

        if not all_contents:
            raise
            return 0, 0

        len_title = sum([len_tok(re.sub('\s+', '', title)) for title in all_titles])
        len_content = sum([len_tok(re.sub('\s+', '', content)) for content in all_contents])

        ratio = len_title / len_content

        last_content = history_arr[-1].get_content().strip() if isinstance(history_arr[-1], TextNode) else history_arr[-1].strip()
        matches = re.findall(pattern, last_content, flags=re.MULTILINE)
        is_last_title = False

        if matches:
            is_last_title = last_content.endswith(matches[-1])

        return ratio, is_last_title

    @staticmethod
    def is_history_title_merge(history_arr: List[BaseNode]|List[str]):
        """title 比例超过0.8 并且 最后一个是title"""
        ratio, is_last_title = ElementChunker.calc_title_ratio(history_arr)
        if ratio > 0.8 and is_last_title:
            return True

        return False

    @staticmethod
    def can_rotate(node: BaseNode|str, history_arr: List[BaseNode]|List[str], min_text_len: int):
        ## 前序内容太短 或者是 标题性质，允许合并进来
        text_len = len_tok(node.get_content()) if isinstance(node, TextNode) else len_tok(node)
        if history_arr and ElementChunker.is_history_title_merge(history_arr):
            return True

        return text_len < min_text_len

    @staticmethod
    def can_rotate_after(node: BaseNode|str, min_text_len: int):
        ## 允许图表合并后续内容条件：文本短，并且非标题
        text_len = len_tok(node.get_content()) if isinstance(node, TextNode) else len_tok(node)
        title_flag = is_md_title(node.get_content()) if isinstance(node, TextNode) else is_md_title(node)
        return text_len < min_text_len and not title_flag

    @staticmethod
    def is_history_title_not_merge(history_arr: List[BaseNode]|List[str]):
        ratio, is_last_title = ElementChunker.calc_title_ratio(history_arr)

        if not is_last_title:
            return False
        if ratio > 0.8 and is_last_title:
            return True
        else:
            return False

    @staticmethod
    def is_standalone_title_text(md_text: str) -> bool:
        lines = [line.strip() for line in (md_text or "").splitlines() if line.strip()]
        if not lines:
            return False
        return all(is_md_title(line) for line in lines)

    @staticmethod
    def can_merge_title_prefix(history_arr: List[BaseNode], next_node: BaseNode) -> bool:
        if not history_arr:
            return False

        if any(node.metadata.get(TYPE_META_CONTENT_TYPE, 'text') != 'text' for node in history_arr):
            return False

        history_text = '\n'.join(
            node.get_content().strip() for node in history_arr if node.get_content().strip()
        )
        if not ElementChunker.is_standalone_title_text(history_text):
            return False

        current_header_path = pick_primary_header_path(
            [node.metadata.get(TYPE_META_HEADER_PATH, '/') for node in history_arr]
        )
        next_header_path = pick_primary_header_path([next_node.metadata.get(TYPE_META_HEADER_PATH, '/')])

        if current_header_path in ('', '/') or next_header_path in ('', '/'):
            return False
        if current_header_path == next_header_path:
            return False

        return next_header_path.startswith(current_header_path)

    @staticmethod
    def has_leading_md_title(md_text: str) -> bool:
        lines = [line.strip() for line in (md_text or "").splitlines() if line.strip()]
        if not lines:
            return False
        return is_md_title(lines[0])

    @staticmethod
    def is_small_heading_text(md_text: str, max_tokens: int) -> bool:
        md_text = (md_text or "").strip()
        if not md_text:
            return False
        if ElementChunker.is_standalone_title_text(md_text):
            return True
        if len_tok(md_text) > max_tokens:
            return False
        return ElementChunker.has_leading_md_title(md_text)

    def can_merge_small_header_path_text(self, history_arr: List[BaseNode], next_node: BaseNode) -> bool:
        if not self.merge_small_header_path_text or not history_arr:
            return False
        if any(node.metadata.get(TYPE_META_CONTENT_TYPE, 'text') != 'text' for node in history_arr):
            return False
        if next_node.metadata.get(TYPE_META_CONTENT_TYPE, 'text') != 'text':
            return False

        history_text = '\n'.join(
            node.get_content().strip() for node in history_arr if node.get_content().strip()
        ).strip()
        if not ElementChunker.is_small_heading_text(history_text, self.small_header_path_text_max_tokens):
            return False

        if not ElementChunker.is_standalone_title_text(history_text) and \
                ElementChunker.is_standalone_title_text(next_node.get_content()):
            return False

        current_header_path = pick_primary_header_path(
            [node.metadata.get(TYPE_META_HEADER_PATH, '/') for node in history_arr]
        )
        next_header_path = pick_primary_header_path([next_node.metadata.get(TYPE_META_HEADER_PATH, '/')])
        return current_header_path != next_header_path

    def _try_append_trailing_title_to_previous(self, arr: List[BaseNode], sub_arr: List[BaseNode]) -> bool:
        if not self.merge_small_header_path_text or not arr or not sub_arr:
            return False
        if any(node.metadata.get(TYPE_META_CONTENT_TYPE, 'text') != 'text' for node in sub_arr):
            return False

        trailing_text = self._get_content_from_md_node(sub_arr).strip()
        if not ElementChunker.is_standalone_title_text(trailing_text):
            return False

        previous_node = arr[-1]
        if previous_node.metadata.get(TYPE_META_CONTENT_TYPE, 'text') != 'text':
            return False

        previous_text = previous_node.get_content().strip()
        if len_tok(previous_text) + len_tok(trailing_text) > self.max_chunk_length:
            return False

        previous_node.set_content('\n\n'.join(part for part in [previous_text, trailing_text] if part))
        return True

    def _merge_node_with_condition(self,allow_by_header,
                                   node, history_arr,
                                   history_arr_chunk_len,
                                   new_node_idx) -> tuple:
        """
        allow_by_header: node 的 header_path 是否 与 history_arr 中的header_path 一致 或者 不一致、允许合并
        """
        ##
        if not allow_by_header:
            flush_cur_node = False
        else:
            flush_cur_node = False
            ## +node 整体长度超过 max_chunk_length， 但是 已有长度 < min_chunk_length, 此时仍然要合并
            ## +node 整体长度超过 max_chunk_length， 但 history_arr 只是标题，此时仍然要合并
            if history_arr_chunk_len < self.min_chunk_length or \
                    ElementChunker.is_history_title_not_merge(history_arr):

                history_arr.append(node)
                history_arr_chunk_len += len_tok(node.get_content())
                flush_cur_node = True

        if history_arr:
            new_node = self._new_node(history_arr, new_node_idx)
            new_node_idx += 1

        if flush_cur_node:
            history_arr = []
            history_arr_chunk_len = 0
        else:
            history_arr = [node]
            history_arr_chunk_len = len_tok(node.get_content())

        return new_node, new_node_idx, history_arr, history_arr_chunk_len


    def _split_table_figure(self, nodes: List[BaseNode]):
        """
        合并条件：需满足2项
        1、header path限制：only_allow_same_header_path 开关，满足
        2、长度限制：或者
            history_arr+node 不超过 max_chunk_length
            history_arr 不小于 min_chunk_length
            ElementChunker.is_history_title(history_arr)
        """

        arr = []
        sub_arr = [] # chunk
        sub_arr_chunk_len = 0

        new_node_idx = 0
        idx = 0
        # for idx, node in enumerate(nodes):

        last_header_path = None

        while idx < len(nodes):
            node = nodes[idx]  ## llamaindex node
            if node.get_content().find("2025E Non-dom. Growth CAGR*| | Gas%| S(2P/1P)") != -1:
                pass

            existing_type = node.metadata.get(TYPE_META_CONTENT_TYPE, '')

            ## 相信markdown的切分
            if existing_type == 'table':
                node_list = [node]
            else:
                node_list = self.split_by_figure_or_table(node,
                                                          add_table_meta=self.append_table,
                                                          min_text_len=self.min_chunk_length,
                                                          min_text_len_after=self.min_chunk_length_after)
            # ## 不相信markdown的切分
            # node_list = self.split_by_figure_or_table(node,
            #                                               add_table_meta=self.append_table,
            #                                               min_text_len=self.min_chunk_length)

            # node_text = re.sub(r'\s+', '', node.text)
            # new_text = ''.join([n_node.get_content() for n_node in node_list])
            # new_text = re.sub(r'\s+', '', new_text)
            # print('\n\n','===='*10)
            # print(node_text)
            # print(new_text)
            # assert node_text == new_text
            n_idx = 0
            while n_idx <len(node_list):
                n_node = node_list[n_idx]

                ## 图/表 单独成段, 不考虑长度
                if n_node.metadata.get(TYPE_META_CONTENT_TYPE) != 'text':
                    node_include = False

                    if n_idx+1 < len(node_list) and node_list[n_idx+1].get_content().find('注：2023 年 6 月 5 日恒生港股通物业服务及管理指数（HSPSM.HI）曾进行成分调整并') !=-1:
                        pass

                    ## 前序内容太短，允许合并进来
                    if ElementChunker.can_rotate(n_node, sub_arr, min_text_len=self.min_chunk_length):
                    # if sub_arr_chunk_len < self.min_chunk_length:
                        sub_arr.append(n_node)
                        sub_arr_chunk_len += len_tok(n_node.get_content())
                        node_include = True

                        if n_idx+1 < len(node_list) and ElementChunker.can_rotate_after(node_list[n_idx+1], self.min_chunk_length_after):
                            # print(f"xxxx {n_idx}, {node_list[n_idx+1]}")

                            node_after = node_list[n_idx + 1]
                            sub_arr.append(node_after)
                            sub_arr_chunk_len += len_tok(node_after.get_content())
                            n_idx +=1

                    if sub_arr:
                        arr.append(self._new_node(sub_arr, new_node_idx))
                        new_node_idx +=1
                        sub_arr = []
                        sub_arr_chunk_len = 0

                    if not node_include:
                        tmp_arr = [n_node]
                        if n_idx + 1 < len(node_list) and ElementChunker.can_rotate_after(node_list[n_idx + 1],
                                                                                          self.min_chunk_length_after):
                            # print(f"xxxx {n_idx}, {node_list[n_idx + 1]}")

                            node_after = node_list[n_idx + 1]
                            tmp_arr.append(node_after)
                            n_idx += 1

                        arr.append(self._new_node(tmp_arr, new_node_idx))
                        new_node_idx += 1

                    last_header_path = None  ## reset

                else:
                    allow_to_merge = sub_arr_chunk_len + len_tok(n_node.get_content()) < self.max_chunk_length
                    ## 未指定 self.only_allow_same_header_path， 允许合
                    ## 指定 self.only_allow_same_header_path，只允许相同的 head_path 的chunk 合
                    allow_by_header = (not self.only_allow_same_header_path or
                                       (self.only_allow_same_header_path and (
                            last_header_path == None or \
                            last_header_path == n_node.metadata.get("header_path", '')))
                                       )
                    if not allow_by_header and self.only_allow_same_header_path and self.allow_title_prefix_header_merge:
                        allow_by_header = ElementChunker.can_merge_title_prefix(sub_arr, n_node)
                    small_header_path_merge = False
                    if not allow_by_header and self.only_allow_same_header_path:
                        small_header_path_merge = self.can_merge_small_header_path_text(sub_arr, n_node)
                        allow_by_header = small_header_path_merge

                    if allow_to_merge & allow_by_header:
                        sub_arr.append(n_node)
                        sub_arr_chunk_len += len_tok(n_node.get_content())
                        last_header_path = node.metadata.get("header_path", '')

                    else:
                        condition_allow_by_header = allow_by_header
                        if small_header_path_merge and not ElementChunker.is_standalone_title_text(
                                '\n'.join(n.get_content().strip() for n in sub_arr if n.get_content().strip())
                        ):
                            condition_allow_by_header = allow_by_header and allow_to_merge

                        new_node, new_node_idx, sub_arr, sub_arr_chunk_len = self._merge_node_with_condition(condition_allow_by_header,
                                                                                                             n_node,
                                                                                                             sub_arr,
                                                                                                             sub_arr_chunk_len,
                                                                                                             new_node_idx)
                        arr.append(new_node)
                        if not sub_arr:
                            last_header_path = None
                        else:
                            last_header_path = node.metadata.get("header_path", '')

                n_idx += 1

            logging.debug(f"idx: {idx}, total: {len(nodes)}")
            idx+=1

        if sub_arr:
            if self._try_append_trailing_title_to_previous(arr, sub_arr):
                return arr
            arr.append(self._new_node(sub_arr, new_node_idx))
            new_node_idx += 1

        return arr

    def _split_with_len(self, nodes: List[BaseNode]):
        arr = []
        sub_arr = []  # chunk
        sub_arr_chunk_len = 0

        new_node_idx = 0
        idx = 0
        # for idx, node in enumerate(nodes):
        last_header_path = None

        while idx < len(nodes):
            node = nodes[idx]  ## llamaindex node
            ## markdown node
            existing_type = node.metadata.get(TYPE_META_CONTENT_TYPE, '')

            # if existing_type == 'table':
            #     node_list = [node]
            # else:
            #     node_list = self.split_by_figure_or_table(node, add_table_meta=self.append_table)
            node_list = [node]

            for n_node in node_list:
                allow_to_merge = sub_arr_chunk_len + len_tok(n_node.get_content()) < self.max_chunk_length
                ## 未指定 self.only_allow_same_header_path， 允许合
                ## 指定 self.only_allow_same_header_path，只允许相同的 head_path 的chunk 合
                allow_by_header = (not self.only_allow_same_header_path or
                                   (self.only_allow_same_header_path and (
                                           last_header_path == None or \
                                           last_header_path == n_node.metadata.get("header_path", '')))
                                   )
                if not allow_by_header and self.only_allow_same_header_path and self.allow_title_prefix_header_merge:
                    allow_by_header = ElementChunker.can_merge_title_prefix(sub_arr, n_node)
                small_header_path_merge = False
                if not allow_by_header and self.only_allow_same_header_path:
                    small_header_path_merge = self.can_merge_small_header_path_text(sub_arr, n_node)
                    allow_by_header = small_header_path_merge

                if allow_to_merge & allow_by_header:
                    sub_arr.append(n_node)
                    sub_arr_chunk_len += len_tok(n_node.get_content())
                    last_header_path = node.metadata.get("header_path", '')

                else:
                    condition_allow_by_header = allow_by_header
                    if small_header_path_merge and not ElementChunker.is_standalone_title_text(
                            '\n'.join(n.get_content().strip() for n in sub_arr if n.get_content().strip())
                    ):
                        condition_allow_by_header = allow_by_header and allow_to_merge

                    new_node, new_node_idx, sub_arr, sub_arr_chunk_len = self._merge_node_with_condition(
                                                                                condition_allow_by_header,
                                                                                n_node,
                                                                                sub_arr,
                                                                                sub_arr_chunk_len,
                                                                                new_node_idx)
                    arr.append(new_node)
                    if not sub_arr:
                        last_header_path = None
                    else:
                        last_header_path = node.metadata.get("header_path", '')

            logging.debug(f"idx: {idx}, total: {len(nodes)}")
            idx += 1

        if sub_arr:
            if self._try_append_trailing_title_to_previous(arr, sub_arr):
                return arr
            arr.append(self._new_node(sub_arr, new_node_idx))
            new_node_idx += 1

        return arr

    def gen_from_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        if self.debug:
            with open('__subparer_results.json', 'w') as f:
                arr = [{"idx": idx, "text": node.get_content(), "meta": node.metadata}
                       for idx, node in enumerate(nodes)]
                json.dump({'arr': arr}, f, ensure_ascii=False)

        if self.is_sep_table_figure:
            return self._split_table_figure(nodes)
        else:
            return self._split_with_len(nodes)

