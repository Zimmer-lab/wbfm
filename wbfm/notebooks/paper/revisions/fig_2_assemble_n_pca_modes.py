#!/usr/bin/env python
# coding: utf-8
"""
Assembles the panels made by fig_2_supp_CCA_n_pca_modes.py into full-page (A4) figures:

    FIG2_5pcs.svg           Main figure 2, same layout as fig2/FIG2.svg, with the 5-PCA-mode panels
                            (panel a is copied from fig2/FIG2.svg, and panel b (pure PCA) from fig2/, since they do not
                            depend on the number of PCA modes)
    FIG_S_n_pca_modes.svg   Supplement: comparison between 3 and 5 neuronal PCA modes
    FIG_S_5pcs_weights.svg  Supplement: remaining CCA weight panels with 5 PCA modes

Each is also exported to pdf and png using Inkscape. Panels are embedded (not linked), at their native physical size
unless scaled, so the font sizes match the individual panels. Every panel is a labeled group, so it can be edited
in Inkscape.

Run from the repo root (a few seconds):
    /home/charles/anaconda3/envs/wbfm/bin/python -I wbfm/notebooks/paper/revisions/fig_2_assemble_n_pca_modes.py
"""

import copy
import re
import subprocess
from pathlib import Path

from lxml import etree

N_PCA_MODES = 5

try:
    BASE_FOLDER = Path(__file__).resolve().parent
except NameError:
    BASE_FOLDER = Path.cwd()
PANEL_FOLDER = BASE_FOLDER / f'fig2_{N_PCA_MODES}pcs'
ORIGINAL_FOLDER = BASE_FOLDER / 'fig2'
OUTPUT_FOLDER = PANEL_FOLDER

SVG_NS = 'http://www.w3.org/2000/svg'
XLINK_NS = 'http://www.w3.org/1999/xlink'
INKSCAPE_NS = 'http://www.inkscape.org/namespaces/inkscape'
NSMAP = {None: SVG_NS, 'xlink': XLINK_NS, 'inkscape': INKSCAPE_NS}
HREF_ATTRIBUTES = ['href', f'{{{XLINK_NS}}}href']
URL_REGEX = re.compile(r"""url\(\s*['"]?#([^)'"]+)['"]?\s*\)""")

# Same style as the panel letters in the original figures
LABEL_STYLE = dict(font_size=8.46667, font_weight='bold', font_family='sans-serif')


def _length_to_mm(length: str) -> float:
    """Converts an svg length to mm; unitless values are px (96 dpi), as in Inkscape"""
    match = re.fullmatch(r'([\d.]+)\s*(px|pt|mm|cm|in)?', length.strip())
    if match is None:
        raise ValueError(f"Could not parse svg length: {length}")
    value, unit = float(match.group(1)), match.group(2) or 'px'
    return value * {'px': 25.4 / 96, 'pt': 25.4 / 72, 'mm': 1, 'cm': 10, 'in': 25.4}[unit]


def _prefix_ids(root, prefix: str):
    """Makes all ids unique within the assembled figure (in place), including the references to them"""
    ids = {el.get('id') for el in root.iter() if isinstance(el.tag, str) and el.get('id')}

    def replace_urls(value):
        return URL_REGEX.sub(lambda m: f'url(#{prefix}{m.group(1)})' if m.group(1) in ids else m.group(0), value)

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue  # Comments
        if el.get('id'):
            el.set('id', prefix + el.get('id'))
        for key, value in el.attrib.items():
            if key in HREF_ATTRIBUTES and value.startswith('#') and value[1:] in ids:
                el.set(key, '#' + prefix + value[1:])
            elif 'url(' in value:
                el.set(key, replace_urls(value))
        if etree.QName(el).localname == 'style' and el.text and 'url(' in el.text:
            el.text = replace_urls(el.text)


def _referenced_ids(el) -> set:
    s = etree.tostring(el).decode()
    return set(URL_REGEX.findall(s)) | set(re.findall(r'href="#([^"]+)"', s))


class FigurePage:
    """An A4 page (in mm) that panels (svg files) are placed on"""

    def __init__(self, width=210, height=297):
        self.root = etree.Element(f'{{{SVG_NS}}}svg', nsmap=NSMAP,
                                  attrib=dict(width=f'{width}mm', height=f'{height}mm', viewBox=f'0 0 {width} {height}',
                                              version='1.1'))
        # White background (not transparent), so the png export looks like the pdf
        etree.SubElement(self.root, f'{{{SVG_NS}}}rect', attrib=dict(x='0', y='0', width=str(width), height=str(height),
                                                                     fill='white', id='background'))
        self.defs = etree.SubElement(self.root, f'{{{SVG_NS}}}defs')
        self.n_panels = 0

    def _clip_group(self, g, x, y, w, h):
        """Clips a group to a rectangle (in page coordinates)

        Note: Inkscape ignores the overflow attribute of nested svg elements, so an explicit clip path is needed
        """
        clip = etree.SubElement(self.defs, f'{{{SVG_NS}}}clipPath', attrib=dict(id=f'clip_{g.get("id")}'))
        etree.SubElement(clip, f'{{{SVG_NS}}}rect', attrib=dict(x=f'{x:.3f}', y=f'{y:.3f}', width=f'{w:.3f}',
                                                              height=f'{h:.3f}'))
        g.set('clip-path', f'url(#clip_{g.get("id")})')

    def _new_group(self, name):
        self.n_panels += 1
        g = etree.SubElement(self.root, f'{{{SVG_NS}}}g', attrib={'id': f'panel{self.n_panels}',
                                                                 f'{{{INKSCAPE_NS}}}label': name})
        return g, f'p{self.n_panels}_'

    def add_panel(self, fname, x, y, scale=1.0, width=None, label=None, label_offset=(-6, 4.5), overflow=None,
                  view_box=None):
        """
        Places an svg file with its top left corner at (x, y), in mm

        By default the panel keeps its native physical size; otherwise pass scale, or a width in mm
        overflow='visible' shows content outside the svg's own canvas, like an Inkscape import does. By default this is
        used for matplotlib panels (which sometimes draw labels there), and not for plotly panels
        view_box overrides the region of the panel that is shown (in its own units), e.g. to include labels that are
        outside its canvas but not others; the size of the panel is scaled accordingly

        Returns the bounding box (x, y, width, height) of the panel, in mm
        """
        fname = Path(fname)
        src = etree.parse(str(fname)).getroot()
        native_w, native_h = _length_to_mm(src.get('width')), _length_to_mm(src.get('height'))
        original_view_box = src.get('viewBox') or f'0 0 {src.get("width")} {src.get("height")}'.replace('px', '').replace('pt', '')
        if overflow is None:
            # An explicit view_box is a crop, so clip to it
            overflow = 'visible' if (src.get('width').strip().endswith('pt') and view_box is None) else 'hidden'
        if view_box is None:
            view_box = original_view_box
        else:
            # Same physical scale as the original canvas
            _, _, w0, h0 = [float(v) for v in original_view_box.split()]
            _, _, w1, h1 = view_box
            native_w, native_h = native_w * w1 / w0, native_h * h1 / h0
            view_box = ' '.join(str(v) for v in view_box)
        if width is not None:
            scale = width / native_w
        w, h = native_w * scale, native_h * scale

        g, prefix = self._new_group(f'{label or ""} {fname.name}'.strip())
        _prefix_ids(src, prefix)
        nested = etree.SubElement(g, f'{{{SVG_NS}}}svg', attrib=dict(x=f'{x:.3f}', y=f'{y:.3f}', width=f'{w:.3f}',
                                                                    height=f'{h:.3f}', viewBox=view_box,
                                                                    overflow=overflow))
        if overflow == 'hidden':
            self._clip_group(g, x, y, w, h)
        for child in src:
            nested.append(copy.deepcopy(child))
        if label is not None:
            self.add_label(label, x + label_offset[0], y + label_offset[1])
        return x, y, w, h

    def add_region_from_svg(self, fname, element_id, region, x, y, label=None, label_offset=(0, 0)):
        """
        Copies one element (e.g. an Inkscape group) from an assembled figure, plus the definitions it uses, and shows
        only the given region = (x0, y0, width, height) of that figure (in its user units, mm for the original figures)
        """
        src = etree.parse(str(fname)).getroot()
        element = src.find(f'.//*[@id="{element_id}"]')
        if element is None:
            raise ValueError(f"Element {element_id} not found in {fname}")
        # Find all definitions that are (recursively) referenced
        all_ids = {el.get('id'): el for el in src.iter() if isinstance(el.tag, str) and el.get('id')}
        to_check, needed = _referenced_ids(element), set()
        while to_check:
            i = to_check.pop()
            if i in needed or i not in all_ids:
                continue
            needed.add(i)
            to_check |= _referenced_ids(all_ids[i])
        needed -= {el.get('id') for el in element.iter() if isinstance(el.tag, str)}

        wrapper = etree.Element(f'{{{SVG_NS}}}svg')
        defs = etree.SubElement(wrapper, f'{{{SVG_NS}}}defs')
        for i in sorted(needed):
            defs.append(copy.deepcopy(all_ids[i]))
        wrapper.append(copy.deepcopy(element))

        g, prefix = self._new_group(f'{label or ""} {Path(fname).name}#{element_id}'.strip())
        _prefix_ids(wrapper, prefix)
        x0, y0, w, h = region
        nested = etree.SubElement(g, f'{{{SVG_NS}}}svg', attrib=dict(x=f'{x:.3f}', y=f'{y:.3f}', width=f'{w:.3f}',
                                                                    height=f'{h:.3f}', viewBox=f'{x0} {y0} {w} {h}',
                                                                    overflow='hidden'))
        self._clip_group(g, x, y, w, h)
        for child in wrapper:
            nested.append(child)
        if label is not None:
            self.add_label(label, x + label_offset[0], y + label_offset[1])

    def add_label(self, text, x, y):
        self.add_text(text, x, y, **LABEL_STYLE)

    def add_text(self, text, x, y, font_size=3.5, font_weight='normal', font_family='sans-serif', anchor='start',
                 font_style='normal'):
        el = etree.SubElement(self.root, f'{{{SVG_NS}}}text',
                              attrib={'x': f'{x:.3f}', 'y': f'{y:.3f}', 'font-size': str(font_size),
                                      'font-weight': font_weight, 'font-family': font_family,
                                      'font-style': font_style, 'text-anchor': anchor})
        el.text = text

    def save(self, fname, export_formats=('pdf', 'png'), png_dpi=300):
        fname = Path(fname)
        etree.ElementTree(self.root).write(str(fname), xml_declaration=True, encoding='utf-8', pretty_print=True)
        print(f"Saved {fname}")
        for fmt in export_formats:
            out = fname.with_suffix(f'.{fmt}')
            cmd = ['inkscape', str(fname), f'--export-type={fmt}', f'--export-filename={out}']
            if fmt == 'png':
                cmd.append(f'--export-dpi={png_dpi}')
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Saved {out}")


def panel(name):
    return PANEL_FOLDER / f'{name}.svg'


# %% Main figure 2, with N_PCA_MODES (same layout as fig2/FIG2.svg)
page = FigurePage()

# Row 1: state transition diagram (copied from the original figure, including the worm cartoons) and example latent spaces
page.add_region_from_svg(ORIGINAL_FOLDER / 'FIG2.svg', 'g34574', region=(0, 0, 79, 62), x=0, y=0)
page.add_label('a', 5.3, 9.7)
row1 = dict(y=5.7, width=38.5)
page.add_panel(ORIGINAL_FOLDER / 'pca-2d.svg', x=82, label='b', **row1)
page.add_panel(panel('cca-continuous-2d'), x=125.5, label='c', **row1)
page.add_panel(panel('cca-binary-2d'), x=169, label='d', **row1)

# Row 2: neuronal variance explained, PCA-CCA similarity, canonical correlations
y = 65
page.add_panel(panel('top_mode_reconstruction_boxplot'), x=10, y=y, label='e')
page.add_panel(panel('mode_dot_product'), x=104, y=y, label='f')
page.add_panel(panel('paired_boxplot_latent_space_combined12345'), x=156, y=y, label='g')

# Row 3: behavioral variance explained, behavioral weights (continuous, discrete)
y = 127
page.add_panel(panel('colored_box_beh_variance_explained'), x=19, y=y, scale=0.85, label='h', label_offset=(-8, 1))
page.add_panel(panel('paired_boxplot_beh_weights'), x=100, y=y, width=47, label='i')
page.add_panel(panel('paired_boxplot_beh_weights_binary'), x=156, y=y, width=47, label='j')

# Row 4: neuronal weights
page.add_panel(panel('paired_boxplot_neural_weights_BOTH'), x=10, y=199, width=196, label='k')

page.save(OUTPUT_FOLDER / f'FIG2_{N_PCA_MODES}pcs.svg')

# %% Supplement: comparison between the number of PCA modes
page = FigurePage()
col_x = [12, 112]

# Row 1: what is gained and lost by keeping more PCA modes
y = 6
page.add_panel(panel('supp_npcs_neuronal_variance_explained'), x=col_x[0], y=y, label='a')
page.add_panel(panel('supp_npcs_behavioral_variance_explained'), x=col_x[1], y=y, label='b')

# Row 2: canonical correlations
y = 69
page.add_panel(panel('supp_npcs_canonical_correlation_CCA'), x=col_x[0], y=y, label='c')
page.add_panel(panel('supp_npcs_canonical_correlation_CCA_Discrete'), x=col_x[1], y=y, label='d')

# Row 3: which PCA modes the CCA modes use
y = 134
page.add_panel(panel('supp_npcs_cca_pca_overlap_CCA'), x=col_x[0], y=y, label='e')
page.add_panel(panel('supp_npcs_cca_pca_overlap_CCA_Discrete'), x=col_x[1], y=y, label='f')
for x, title in zip(col_x, ['Continuous behaviors', 'Discrete behaviors']):
    page.add_text(title, x + 38, y + 1.5, font_size=4.2, anchor='middle')

# Row 4: robustness of the neuronal weights
y = 191
page.add_panel(panel('supp_npcs_weight_similarity'), x=col_x[0], y=y, label='g')
page.add_panel(panel('supp_npcs_median_weights_scatter'), x=72, y=y, label='h')

page.save(OUTPUT_FOLDER / 'FIG_S_n_pca_modes.svg')

# %% Supplement: remaining weight panels, with N_PCA_MODES
page = FigurePage()

# Row 1: neuronal weights of mode 2
page.add_panel(panel('paired_boxplot_neural_weights2_BOTH'), x=10, y=6, width=196, label='a')

# Row 2: behavioral weights of modes 2 and 3
y = 92
page.add_panel(panel('paired_boxplot_beh_weights2'), x=12, y=y, label='b')
page.add_panel(panel('paired_boxplot_beh_weights_binary2'), x=78, y=y, label='c')
page.add_panel(panel('paired_boxplot_beh_weights3'), x=144, y=y, label='d')

# Row 3: neuronal variance explained per neuron, behavioral variance explained by discrete CCA
y = 184
page.add_panel(panel('variance_explained_by_cca_cumulative'), x=12, y=y, label='e')
# Crop to the row labels (outside of the matplotlib canvas), without the redundant title and group label
page.add_panel(panel('colored_box_beh_variance_explained_binary'), x=100, y=y, scale=0.85, label='f',
               view_box=(-70, 0, 245, 316.8))

page.save(OUTPUT_FOLDER / f'FIG_S_{N_PCA_MODES}pcs_weights.svg')
