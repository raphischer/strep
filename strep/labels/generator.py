import os
import base64
import datetime
import json

from PIL import Image
import numpy as np
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import black, white
import fitz # install via PyMuPDF
import qrcode

from strep.util import lookup_meta

C_SIZE = (1560, 2411)

POS_GENERAL = {
    # infos that are directly taken from summary via keys
    "model":       (.04,  .855, 'drawString',        80, '-Bold'),
    "task":        (.04,  .815, 'drawString',        80, ''),
    "environment": (.04,  .42,  'drawString',        65, ''),
    "dataset":     (.95,  .815, 'drawRightString',   80, ''),
}

POS_RATINGS = { char: (.66, y) for char, y in zip('ABCDE', reversed(np.linspace(.461, .727, 5))) }

POS_METRICS = {
    'upper_left': {
        'icon':  (0.25, 0.33)
    },
    'upper_right': {
        'icon':  (0.75, 0.33)
    },
    'lower_left': {
        'icon':  (0.25, 0.137)
    },
    'lower_right': {
        'icon':  (0.75, 0.137)
    },
}

PARTS_DIR = os.path.join(os.path.dirname(__file__), "label_design", "graphic_parts")

ICONS = { f.split('_0.png')[0]: os.path.join(PARTS_DIR, f.replace('_0.', '_$.')) for f in os.listdir(PARTS_DIR) if f.endswith('_0.png') }

def place_relatively(canvas, rel_x, rel_y, draw_method, content, fontstyle='', font_size=None):
    image = 'Image' in draw_method
    draw_method = getattr(canvas, draw_method)
    x, y = int(C_SIZE[0] * rel_x), int(C_SIZE[1] * rel_y)
    if image:
        img = Image.open(content)
        draw_method(content, x - img.width // 2, y - img.height // 2)
    else:
        canvas.setFont('Helvetica' + fontstyle, font_size)
        if isinstance(content, dict) and 'name' in content:
            content = content['name']
        draw_method(x, y, content)


def format_power_draw_sources(summary):
    sources = 'Sources:'
    for key, vals in summary['power_draw_sources'].items():
        if len(vals) > 0:
            sources += f' {key},'
    return sources[:-1]


def create_qr(url):
    qr = qrcode.QRCode(
        version=1, box_size=1, border=0,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img


def draw_qr(canvas, qr, x, y, width):
    qr_pix = np.array(qr)
    width //= qr_pix.shape[0]
    for (i, j), v in np.ndenumerate(qr_pix):
        if v:
            canvas.setFillColor(white)
        else:
            canvas.setFillColor(black)
        canvas.rect(x + (i * width), y + int(width * qr_pix.shape[0]) - ((j + 1) * width), width, width, fill=1, stroke=0)


def find_icon(metric_key, metric, icons):
    metric_group = metric['group'].lower()
    # check for icon info in meta
    if 'icon' in metric and metric['icon'] in icons:
        return icons[metric['icon']]
    # check for exact name match
    for key, path in icons.items():
        if key == metric_key:
            return path
    # check for exact group match
    for key, path in icons.items():
        if key == metric_group:
            return path
    # check for similar name match
    for key, path in icons.items():
        if key in metric_key:
            return path
    # check for similar group match
    for key, path in icons.items():
        if key in metric_group:
            return path
    # TODO implement option to pass a mapping dictionary?
    return next(iter(icons.values()))


def select_top_metrics(metrics):
    # Capture original insertion order and indexes
    ordered_names = list(metrics.keys())
    index_map = {name: idx for idx, name in enumerate(ordered_names)}
    
    groups = {}
    for name, (group, weight) in metrics.items():
        if group not in groups:
            groups[group] = []
        # Store (-weight, original index, name) for sorting
        groups[group].append( (-weight, index_map[name], name) )
    
    # Sort each group's metrics by: highest weight first, then earliest index
    for group in groups:
        groups[group].sort()
    
    # Collect top metric from each group (highest weight/earliest index)
    top_candidates = []
    for group, metrics_list in groups.items():
        top_metrics = metrics_list[0]  # Take highest weight/earliest index
        top_candidates.append( (top_metrics[0], top_metrics[1], top_metrics[2], group) )
    
    # Sort top_candidates by weight (descending) and then original index (ascending)
    top_candidates.sort()
    
    selected = []
    group_counts = {group: 0 for group in groups.keys()}
    
    # Select metrics while maintaining group diversity
    for weight_neg, idx, name, group in top_candidates:
        if len(selected) >= 4:
            break
        if group_counts[group] < 1:  # Ensure max 1 per group
            selected.append( (idx, name) )
            group_counts[group] += 1
    
    # Sort selected metrics by their original insertion order (index)
    selected.sort(key=lambda x: x[0])
    return [name for idx, name in selected]


class PropertyLabel(fitz.Document):

    def __init__(self, summary, meta, unit_fmt, custom=None):
        # check if custom badge positions are wanted
        try:
            with open(os.path.join(custom, 'label_map.json'), 'r') as jf:
                metric_map = json.load(jf)
        except Exception:
            properties = {prop: (vals['group'], float(vals['weight'])) for prop, vals in meta.items()}
            most_important = select_top_metrics(properties)
            metric_map = {pos: prop for pos, prop in zip(POS_METRICS.keys(), most_important)}

        icons = ICONS.copy()
        bg = os.path.join(PARTS_DIR, "bg.png")
        # check for custom icons and background
        if custom is not None:
            for fname in os.listdir(custom):
                if '_0.png' in fname:
                    icons[fname.split('_0.png')[0]] = os.path.join(custom, fname.replace('_0.png', '_$.png'))
            if os.path.isfile(os.path.join(custom, "bg.png")):
                bg = os.path.join(custom, "bg.png")
        canvas = Canvas("result.pdf", pagesize=C_SIZE)
        # background
        place_relatively(canvas, 0.5, 0.5, 'drawInlineImage', bg)
        # rating
        frate = 'ABCDE'[summary['compound_rating']]
        pos = POS_RATINGS[frate]
        place_relatively(canvas, pos[0], pos[1], 'drawInlineImage', os.path.join(PARTS_DIR, f"rating_{frate}.png"))
        # qr codes
        if 'url' in summary['model']:
            qr = create_qr(summary['model']['url'])
            draw_qr(canvas, qr, 0.84 * C_SIZE[0], 0.725 * C_SIZE[1], 175)
        qr_strep = create_qr('https://github.com/raphischer/strep')
        draw_qr(canvas, qr_strep, 0.84 * C_SIZE[0], 0.896 * C_SIZE[1], 175)

        # Add stroke to make stronger letters
        canvas.setFillColor(black)
        canvas.setLineWidth(3)
        canvas.setStrokeColor(black)
        text=canvas.beginText()
        text.setTextRenderMode(2)
        canvas._code.append(text.getCode())

        # general text
        for key, (rel_x, rel_y, draw_method, fsize, style) in POS_GENERAL.items():
            place_relatively(canvas, rel_x, rel_y, draw_method, summary[key], style, fsize)
        today = datetime.date.today()
        place_relatively(canvas, .95,  .87, 'drawRightString', f"Issued {today.strftime('%B')[:3]} '{today.strftime('%Y')[-2:]}", '', 56)
        if 'url' in summary['model']:
            place_relatively(canvas, .82,  .776, 'drawRightString', f'Scan for further information', '', 56)

        # rated pictograms & values
        for location, positions in POS_METRICS.items():
            if location in metric_map:
                metric_key = metric_map[location]
                value = summary[metric_key]
                # print icon
                icon = find_icon(metric_key, meta[metric_key], icons)
                rating = int(summary[f'{metric_key}_rating'])
                icon = icon.replace('_$.', f'_{rating}.')
                rel_x, rel_y = positions['icon']
                # print texts
                # TODO improve this by looking at the absolute height of the placed icon
                place_relatively(canvas, rel_x, rel_y, 'drawInlineImage', icon)
                try:
                    fmt_val, fmt_unit = unit_fmt.reformat_value(value, lookup_meta(meta, metric_key, key="unit"))
                    formatted = f"{fmt_val} {fmt_unit}"
                except Exception:
                    formatted = 'N.A.'
                place_relatively(canvas, rel_x, rel_y - 0.08, 'drawCentredString', formatted, '', 56)
                place_relatively(canvas, rel_x, rel_y - 0.11, 'drawCentredString', lookup_meta(meta, metric_key), '', 56)
        
        super().__init__(stream=canvas.getpdfdata(), filetype='pdf')
    
    def to_encoded_image(self):
        label_bytes = self.load_page(0).get_pixmap().tobytes()
        base64_enc = base64.b64encode(label_bytes).decode('ascii')
        return 'data:image/png;base64,{}'.format(base64_enc)
