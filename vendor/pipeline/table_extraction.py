# -*- coding: utf-8 -*-
"""Conservative pipe tables; source text is never normalized or modified."""
# 中文说明：本模块实现"保守的竖线表格（pipe table）识别与提取"。
# 设计原则：
#   1. 保守识别——只提取高置信度的表格，拿不准的一律标记为
#      ambiguous（模糊）而非强行解析；
#   2. 源文本零改动——绝不规范化/修改原文，所有表格的 text/span
#      均直接来自原始字符串的精确切片，保证可回溯定位。
 
from collections import Counter
from datetime import date
import re
 
from markdown_it import MarkdownIt
 
# 版本号：用于诊断报告的版本追踪，格式改变时递增
VERSION = 'pipe-table-v1'
# 参数集合：所有启发式阈值集中在此，便于调参与复现实验
PARAMETERS = {
    'min_columns': 2,               # 表格至少要有 2 列才被接受
    'min_data_rows': 2,             # 表头之外至少要有 2 行数据
    'typed_column_min_ratio': 0.8,  # 某列中"带类型值"（数字/日期等）占比需 ≥ 80%
    'typed_column_min_values': 2,   # 该列带类型的值至少要有 2 个
    'require_consistent_columns': True,   # 每行列数必须一致
    'allow_headerless': False,      # 不接受"无表头"的表格
    'allow_multiline_cells': False, # 不接受跨多行的单元格
    'allow_pure_text_tables': False,# 不接受纯文本（无类型列）表格
    'mutate_source_text': False,    # 严禁修改源文本（自检开关/声明）
}
 
 
def _pipes(line):
    # 计算一行中所有"有效竖线 |"的列位置。
    # 处理转义：反斜杠 \| 会被 Markdown 转义成字面量竖线，
    # 因此只有当竖线前面有偶数个连续反斜杠时才视为"分隔符竖线"。
    # 返回：竖线所在下标的列表。
    positions = []
    slashes = 0  # 当前连续反斜杠的计数（遇非反斜杠即清零）
    for i, char in enumerate(line):
        if char == '|' and slashes % 2 == 0:
            # 偶数个反斜杠 => 竖线未被转义，是真正的分隔符
            positions.append(i)
        # 更新反斜杠计数：遇到 '\' 递增，其他字符清零
        slashes = slashes + 1 if char == '\\' else 0
    return positions
 
 
def _cells(line, borders):
    # 按竖线把一行切成单元格（不 strip，保留原文）。
    # borders 是二元组 (left, right)：
    #   left  = True 表示行首有边框竖线（即该竖线左边的字符全是空白）
    #   right = True 表示行尾有边框竖线（该竖线右边全是空白）
    # 边框竖线不属于任何单元格，会被剔除；内部竖线才是分隔符。
    line = line.rstrip('\r\n')  # 去掉行尾换行符（仅本地副本，不影响源文本）
    positions = _pipes(line)    # 所有有效竖线的位置
    left, right = borders
    a, b = 0, len(line)         # 单元格区域的起止下标（默认整行）
    if left:
        # 行首应有边框竖线；若没有竖线或竖线前有非空白内容 => 边界歧义
        if not positions or line[:positions[0]].strip():
            raise ValueError('AMBIGUOUS_BORDER')
        a = positions.pop(0) + 1  # 起点移到首竖线之后
    if right:
        # 行尾应有边框竖线；若没有竖线或末竖线后有非空白内容 => 边界歧义
        if not positions or line[positions[-1] + 1:].strip():
            raise ValueError('AMBIGUOUS_BORDER')
        b = positions.pop()       # 终点移到末竖线处
    # 切分点序列：[a-1, 内部各竖线, b]
    # 相邻两个切分点之间即一个单元格（保留原始空白，不做 strip）
    cuts = [a - 1, *positions, b]
    return [line[cuts[i] + 1:cuts[i + 1]] for i in range(len(cuts) - 1)]
 
 
def _value_type(value):
    # 单元格值类型推断（用于判断列是否"稳定带类型"）。
    # 返回 'date' / 'month' / 'number' / 'text' 之一。
    value = value.strip()
    # 1) ISO 日期：YYYY-MM-DD，且必须是真实合法日期（如 2026-02-30 不算）
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        try:
            date.fromisoformat(value)  # 借助标准库校验合法性
            return 'date'
        except ValueError:
            return 'text'              # 格式对但日期不存在 => 视为文本
    # 2) 年月：YYYY-MM（月份限定 01-12）
    if re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])', value):
        return 'month'
    # 3) 数字：可选货币符号（$€£¥）+ 可选正负号 + 整数/千分位/小数/
    #    纯小数（.5）+ 可选科学计数法 + 可选百分号
    if re.fullmatch(r'[$€£¥]?[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?', value):
        return 'number'
    # 4) 其余一律视为文本
    return 'text'
 
 
def _old_parse(tokens):
    # 复用旧的（vendored）解析器：从 markdown-it 的 token 流中提取表格。
    # 保持与旧版 t4_splitter 的行为兼容，此处仅做转发。
    from t4_splitter._vendor.dc_parse.chunking.element_chunking import ElementChunker
    return ElementChunker.extract_tables_from_tokens(tokens)
 
 
def extract_tables_with_diagnostics(text):
    # 主入口：从 Markdown 文本中提取表格。
    # 返回 (tables, report)：
    #   tables — 接受的表格列表，每张表含解析结果 + 原文精确切片（start/end/text）
    #            + 表头与每行的逐行 span（便于回溯高亮）；
    #   report — 诊断报告：各类计数 + 每个"未接受"片段的原因与判定。
    #
    # 识别分两条路径：
    #   A. 标准 Markdown 表格（带 |---| 分隔行）——由 markdown-it 识别；
    #   B. 无分隔行的竖线表格——靠下方一组保守启发式判断。
 
    # parser：启用 table 扩展的 CommonMark 解析器（用于路径 A/B 的语法解析）
    parser = MarkdownIt('commonmark').enable('table')
    # inline：只用于渲染行内片段（粗体/链接等），供单元格值规整化
    inline = MarkdownIt()
 
    # ---- 行偏移表：把"行号"映射到"源文本中的字符下标" ----
    lines = text.splitlines(keepends=True)  # 保留换行符，保证偏移精确
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    # 现在 offsets[i] = 第 i 行在 text 中的起始下标，
    # offsets[i+1] = 第 i 行结束（含换行符）后的下标。
 
    tokens = parser.parse(text)
    # blocked：已被"代码块/HTML 块/标题/已识别表格"占用的行号集合，
    # 后续启发式扫描要跳过这些行（避免把代码块里的 | 当表格）。
    blocked = set()
    tables, diagnostics = [], []
 
    # ===== 路径 A：标准 Markdown 表格（有分隔行）=====
    for i, token in enumerate(tokens):
        if token.map is None:
            continue  # 无法定位行号的 token 跳过
        lo, hi = token.map  # 该 token 覆盖的行区间 [lo, hi)
        # 代码围栏 / 缩进代码块 / HTML 块 / 标题：这些区域禁止表格识别
        if token.type in ('fence', 'code_block', 'html_block', 'heading_open'):
            blocked.update(range(lo, hi))
        if token.type != 'table_open':
            continue  # 只关心表格开始 token
 
        # 找到与之配对的 table_close
        j = i + 1
        while j < len(tokens) and tokens[j].type != 'table_close':
            j += 1
        if j == len(tokens):
            raise ValueError('Unclosed Markdown table')  # 结构损坏，直接报错
 
        # 交给旧解析器提取 header/rows（复用既有实现，保证行为一致）
        parsed = _old_parse(tokens[i:j + 1])[0]
        # 用字符偏移精确切出原文片段（绝不改动原文）
        a, b = offsets[lo], offsets[hi]
        tables.append({**parsed, 'start': a, 'end': b, 'text': text[a:b],
                       'detected_by': 'markdown',  # 识别路径标记
                       'recognition_reasons': ['STANDARD_MARKDOWN'],
                       # 表头 span：表格首行（分隔行之前的表头行）
                       'header_span': {'start': offsets[lo], 'end': offsets[lo + 1]},
                       # 数据行 span：跳过表头行(lo)与分隔行(lo+1)，从 lo+2 起
                       'row_spans': [{'start': offsets[k], 'end': offsets[k + 1]} for k in range(lo + 2, hi)]})
        # 已被标准表格占用的行，启发式扫描不再处理
        blocked.update(range(lo, hi))
 
    # ===== 路径 B：无分隔行的竖线表格（启发式，逐段扫描）=====
    i = 0
    while i < len(lines):
        # 跳过：已占用的行、或本行没有竖线（不可能开表格）
        if i in blocked or not _pipes(lines[i]):
            i += 1
            continue
        # 向下贪心扩展：收集连续的、未占用、且都含竖线的行 => 候选表格段
        j = i + 1
        while j < len(lines) and j not in blocked and _pipes(lines[j]):
            j += 1
        a, b = offsets[i], offsets[j]  # 该段在源文本中的精确切片范围
        group = lines[i:j]
 
        # ---- 对候选段做保守判定 ----
        # reason = None 表示通过所有检查；decision 默认 'rejected'（拒绝），
        # 检查不过时置 'ambiguous'（拿不准，宁可放过也不硬解析）。
        reason = None
        decision = 'rejected'
        raw = []     # 未 strip 的原始单元格（每行一个列表）
        stable = []  # 判定为"稳定带类型"的列下标
 
        if len(group) < PARAMETERS['min_data_rows'] + 1:
            # 行数不足（1 表头 + min_data_rows 数据行都凑不齐）
            reason = 'INSUFFICIENT_ROWS'
        elif any('`' in line for line in group):
            # 含行内代码（反引号）的单元格语法不支持 => 直接拒绝
            reason = 'UNSUPPORTED_CELL_SYNTAX'
        else:
            # 以首行为模板确定左右边框样式（首行有边框则后续行必须同样有）
            head = group[0].rstrip('\r\n')
            pipes = _pipes(head)
            borders = (not head[:pipes[0]].strip(),   # 左边框：首竖线前全空白
                       not head[pipes[-1] + 1:].strip())  # 右边框：末竖线后全空白
            try:
                # 按统一边框样式切分每一行（边框歧义会抛 AMBIGUOUS_BORDER）
                raw = [_cells(line, borders) for line in group]
            except ValueError as error:
                reason = str(error)  # 记录异常原因（如 AMBIGUOUS_BORDER）
            if reason is None:
                # 去除首尾空白得到规整单元格（仅用于判断，不回写原文）
                cells = [[v.strip() for v in row] for row in raw]
                width = len(cells[0])  # 以首行为准的列数
                if width < PARAMETERS['min_columns']:
                    reason = 'INSUFFICIENT_COLUMNS'          # 列数太少
                elif any(len(row) != width for row in cells):
                    reason = 'COLUMN_COUNT_MISMATCH'         # 各行列数不一致
                elif any(row == cells[0] for row in cells[1:]):
                    # 某数据行与表头完全相同 => 大概率是重复贴的表头，
                    # 无法确定真正的表头在哪 => 标记为 ambiguous
                    reason, decision = 'REPEATED_HEADER', 'ambiguous'
                elif not all(cells[0]) or not any(
                        _value_type(v) == 'text' and len(v) <= 80 and len(v.split()) <= 8
                        and any(c.isalpha() for c in v) for v in cells[0]):
                    # 表头质量检查：表头不能有空单元格，且至少要有一个
                    # "像表头"的单元格（文本型、≤80字符、≤8个词、含字母）
                    reason, decision = 'AMBIGUOUS_HEADER', 'ambiguous'
                else:
                    # ---- 检查是否存在"稳定带类型"的列 ----
                    # 某列若满足：非文本类型值 ≥ min_values 个，
                    # 且占该列非空值的比例 ≥ min_ratio，则视为稳定类型列。
                    # 这是判断"这真的是一张数据表而非对齐排版文本"的核心信号。
                    for column in range(width):
                        values = [row[column] for row in cells[1:] if row[column]]
                        kinds = Counter(_value_type(v) for v in values)
                        typed = max((n for kind, n in kinds.items() if kind != 'text'), default=0)
                        if typed >= PARAMETERS['typed_column_min_values'] and typed / len(values) >= PARAMETERS['typed_column_min_ratio']:
                            stable.append(column)
                    if not stable:
                        # 一个类型稳定的列都没有 => 允许纯文本表格为 False，
                        # 保守起见标记为 ambiguous（不硬解析）
                        reason, decision = 'NO_STABLE_TYPED_COLUMN', 'ambiguous'
                    elif not any(_value_type(cells[0][column]) == 'text'
                                 or _value_type(cells[0][column]) in ('date', 'month')
                                 or re.fullmatch(r'(?:19|20)\d{2}', cells[0][column])
                                 for column in stable):
                        # 表头与稳定列的呼应检查：至少某个稳定列的表头
                        # 要像"字段名"（文本型 / 日期型 / 19xx 或 20xx 年份）。
                        # 若稳定列的表头全是数字/百分比等，则表头存疑 => ambiguous。
                        reason, decision = 'AMBIGUOUS_HEADER', 'ambiguous'
 
        if reason is None:
            # ===== 全部检查通过：构造虚拟分隔行并做"往返校验" =====
            # 说明：源文本本身没有 |---| 分隔行，markdown-it 无法直接解析。
            # 这里只在本地的虚拟字符串中临时补一条分隔行喂给解析器，
            # —— 源文本本身自始至终未被修改（mutate_source_text=False）。
            # 分隔行的左右边框与首行保持一致。
            separator = ('| ' if borders[0] else '') + ' | '.join(['---'] * len(raw[0])) + (' |' if borders[1] else '')
            virtual = group[0].rstrip('\r\n') + '\n' + separator + '\n' + ''.join(group[1:])
            # 用完整解析器解析虚拟文本
            parsed_tables = _old_parse(parser.parse(virtual))
            # 期望结果：对每个原始单元格做行内渲染（粗体/链接等）后 strip
            expected = [[inline.renderInline(cell.strip()).strip() for cell in row] for row in raw]
            # 往返校验：解析器输出必须与期望完全一致且恰好产出一张表。
            # 不一致说明存在本启发式未覆盖的语法（如转义、嵌套标记），
            # 宁可放弃也不给出错误结果。
            if len(parsed_tables) != 1 or [parsed_tables[0]['header'], *parsed_tables[0]['rows']] != expected:
                reason = 'PARSE_ROUNDTRIP_MISMATCH'
            else:
                # 正式收录：解析结果 + 原始未规整单元格 + 精确原文切片 + 逐行 span
                tables.append({**parsed_tables[0], 'raw_header': raw[0], 'raw_rows': raw[1:],
                               'start': a, 'end': b, 'text': text[a:b],
                               'header_span': {'start': offsets[i], 'end': offsets[i + 1]},
                               'row_spans': [{'start': offsets[k], 'end': offsets[k + 1]} for k in range(i + 1, j)],
                               'detected_by': 'pipe_without_separator',  # 识别路径 B 标记
                               'recognition_reasons': ['CONSISTENT_COLUMNS', 'HEADER_PRESENT', 'STABLE_TYPED_COLUMN'],
                               'typed_columns': stable})  # 记录哪些列是稳定类型列
 
        if reason:
            # 未通过的候选段：写入诊断（含精确 span、判定、原因码），
            # 上层可据此提示用户或做人工复核
            diagnostics.append({'start': a, 'end': b, 'decision': decision, 'reason': reason})
        # 无论接受与否，整段都已处理完，跳到段尾继续扫描
        i = j
 
    # ---- 汇总输出 ----
    # 按出现位置排序，并分配稳定的表编号 table-1, table-2, ...
    tables.sort(key=lambda t: t['start'])
    for index, table in enumerate(tables, 1):
        table['table_id'] = f'table-{index}'
        # 一致性断言：text 必须与源文本切片完全一致（零改动保证）
        assert text[table['start']:table['end']] == table['text']
        # 一致性断言：行 span 数量必须等于解析出的数据行数
        assert len(table['row_spans']) == len(table['rows'])
 
    # 按识别路径统计数量，组装诊断报告
    kinds = Counter(t['detected_by'] for t in tables)
    report = {'version': VERSION, 'parameters': dict(PARAMETERS),
              'accepted': len(tables),                       # 接受的表格总数
              'markdown': kinds['markdown'],                 # 路径 A：标准表格数
              'pipe_without_separator': kinds['pipe_without_separator'],  # 路径 B 数
              'ambiguous': sum(d['decision'] == 'ambiguous' for d in diagnostics),  # 拿不准数
              'rejected': sum(d['decision'] == 'rejected' for d in diagnostics),    # 拒绝数
              'diagnostics': diagnostics}                    # 全部未接受片段的明细
    return tables, report