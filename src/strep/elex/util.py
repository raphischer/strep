from plotly import colors
from plotly.validators.scatter.marker import SymbolValidator
from dash import html
import numpy as np


ENV_SYMBOLS = [SymbolValidator().values[i] for i in range(0, len(SymbolValidator().values), 12)]
RATING_COLORS = ['rgb(99,155,48)', 'rgb(184,172,43)', 'rgb(248,184,48)', 'rgb(239,125,41)', 'rgb(229,36,33)', 'rgb(36,36,36)']
RATING_COLOR_SCALE = colors.make_colorscale([RATING_COLORS[idx] for idx in range(5)])
RATING_COLOR_SCALE_REV = colors.make_colorscale([RATING_COLORS[4-idx] for idx in range(5)])
PATTERNS = ["", "/", ".", "x", "-", "\\", "|", "+", "."]
RATING_MEANINGS = 'ABCDE'


def hex_to_alpha(hex, alpha):
    hex = hex.lstrip('#')
    if len(hex) == 6:
        hex += 'FF'
    r, g, b, a = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4, 6))
    a = f"{alpha:3.1f}"
    return f'rgba({r},{g},{b},{a})'


def rgb_to_rgba(rgb, alpha):
    return rgb.replace('rgb', 'rgba').replace(')', f',{alpha:3.1f})')
    


# def summary_to_str(summary, rating_mode):
#     environment = f"({summary['environment']} Environment)"
#     ret_str = [f'Name: {summary["name"]:17} {environment:<34} - Final Rating {final_rating}']
#     for key, val in summary.items():
#         if isinstance(val, dict) and "value" in val:
#             if val["value"] is None:
#                 value, index = f'{"n.a.":<13}', "n.a."
#             else:
#                 value, index = f'{val["value"]:<13.3f}', f'{val["index"]:4.2f}'
#             ret_str.append(f'{AXIS_NAMES[key]:<30}: {value} - Index {index} - Rating {val["rating"]}')
#     full_str = '\n'.join(ret_str)
#     return full_str


def summary_to_html_tables(summary, properties, unit_fmt):
    # general info
    final_rating = f"{summary['compound_index']:5.3f} ({RATING_MEANINGS[summary['compound_rating']]})"
    info_header = [
        html.Thead(html.Tr([html.Th("Task"), html.Th("Model Name"), html.Th("Environment"), html.Th("Final Rating")]))
    ]
    ds_name = summary['dataset']['name'] if isinstance(summary['dataset'], dict) else summary['dataset']
    task = f"{summary['task']} on {ds_name}"
    mname = summary['model']['name'] if isinstance(summary['model'], dict) else summary['model']
    info_row = [html.Tbody([html.Tr([html.Td(field) for field in [task, mname, summary['environment'], final_rating]])])]

    # properties
    properties_header = [
        html.Thead(html.Tr([html.Th("Property"), html.Th("Value"), html.Th("Index"), html.Th("Rating"), html.Th("Weight")]))
    ]
    properties_rows = []
    for prop, meta in properties.items():
        name = meta["name"] if "name" in meta else prop
        if np.isnan(summary[prop]):
            fmt_val, fmt_unit = 'N.A.', meta["unit"]
        else:
            fmt_val, fmt_unit = unit_fmt.reformat_value(summary[prop], meta["unit"])
        index, rating = summary[f'{prop}_index'], summary[f'{prop}_rating']
        table_cells = [f'{name} {fmt_unit}', fmt_val, f'{index:5.3f}'[:5], rating, f'{meta["weight"]:3.2f}']
        properties_rows.append(html.Tr([html.Td(field) for field in table_cells]))

    model = info_header + info_row
    metrics = properties_header + [html.Tbody(properties_rows)]
    return model, metrics


def toggle_element_visibility(n1, is_open):
    if n1:
        return not is_open
    return is_open

