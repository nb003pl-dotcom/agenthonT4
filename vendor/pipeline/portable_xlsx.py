"""Portable, typed XLSX export; source data never becomes an Excel formula."""
from datetime import date
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


def export_xlsx(spec, output, previews=None):
    if previews:
        raise ValueError('Portable exporter does not generate PNG previews; open tables.xlsx instead')
    workbook = Workbook()
    workbook.remove(workbook.active)
    for table in spec['tables']:
        sheet = workbook.create_sheet(table['sheet'])
        for row_index, row in enumerate(table['cells'], 1):
            for column_index, item in enumerate(row, 1):
                value = item['value']
                if item['kind'] == 'date':
                    value = date.fromisoformat(value)
                cell = sheet.cell(row_index, column_index, value)
                if item['kind'] == 'text':
                    cell.data_type = 's'
                cell.number_format = item['format']
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                if row_index == 1:
                    cell.font = Font(bold=True, color='FFFFFF')
                    cell.fill = PatternFill('solid', fgColor='24476B')
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = table['range']
        for index in range(1, table['column_count'] + 1):
            sheet.column_dimensions[get_column_letter(index)].width = 24
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    workbook.close()
