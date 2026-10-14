"""Normalize Markdown for retrieval while retaining visible document content.

The extracted source stays untouched. Link destinations and HTML attributes
have no retrieval value; headings, labels, lists and table cell associations do.
"""
import re

import markdown
from bs4 import BeautifulSoup, Comment, Doctype, NavigableString, Tag

_FENCED_CODE = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?^ {0,3}\1[^\n]*$")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)")
_HTML_TAGS = set("a abbr b blockquote br caption center code col colgroup dd del div dl dt em figcaption figure h1 h2 h3 h4 h5 h6 hr i img li ol p pre s section small span strong sub sup table tbody td th thead tfoot tr u ul script style".split())
_TAG = re.compile(r"</?([A-Za-z][\w:-]*)\b[^>]*>")
_REFERENCE = re.compile(r"(?m)^ {0,3}\[([^]\n]+)\]:[^\n]*(?:\n {1,3}[\"'(][^\n]*[\"')])?\n?")


def transform_markdown_prose(text: str, transform) -> str:
    """Apply prose cleanup without changing literal inline/fenced code bytes."""
    protected = []

    def protect(match):
        token = f'\ufdd0{len(protected)}\ufdd1'
        if token in text:
            raise ValueError('Markdown protection token collision')
        protected.append(match.group(0))
        return token

    value = _INLINE_CODE.sub(protect, _FENCED_CODE.sub(protect, text))
    value = transform(value)
    for i, original in enumerate(protected):
        value = value.replace(f'\ufdd0{i}\ufdd1', original)
    return value


def _table(node: Tag) -> str:
    """Expand spans so each cell remains associated with its row and column."""
    rows, pending = [], {}
    trs = [tr for tr in node.find_all('tr') if tr.find_parent('table') is node]
    has_header = bool(trs and trs[0].find('th', recursive=False))
    for tr in trs:
        cells, current, col = tr.find_all(['td', 'th'], recursive=False), {}, 0
        for position, (value, remaining) in list(pending.items()):
            current[position] = value
            if remaining == 1:
                del pending[position]
            else:
                pending[position] = (value, remaining - 1)
        for cell in cells:
            while col in current:
                col += 1
            value = re.sub(r'\s+', ' ', _render(cell)).strip().replace('|', r'\|')
            colspan = max(1, int(cell.get('colspan', 1)))
            rowspan = max(1, int(cell.get('rowspan', 1)))
            for position in range(col, col + colspan):
                current[position] = value
                if rowspan > 1:
                    pending[position] = (value, rowspan - 1)
            col += colspan
        if current:
            rows.append(current)
    if not rows:
        return '\n\n' + node.get_text(' ', strip=True) + '\n\n'
    width = max(max(row) for row in rows) + 1
    values = [[row.get(i, '') for i in range(width)] for row in rows]
    if not has_header:
        values.insert(0, [f'列{i+1}' for i in range(width)])
    lines = ['| ' + ' | '.join(row) + ' |' for row in values]
    lines.insert(1, '| ' + ' | '.join(['---'] * width) + ' |')
    caption = node.find('caption', recursive=False)
    prefix = _render(caption).strip() + '\n\n' if caption else ''
    return '\n\n' + prefix + '\n'.join(lines) + '\n\n'


def _render(node) -> str:
    if isinstance(node, (Comment, Doctype)):
        return ''
    if isinstance(node, NavigableString):
        return str(node)
    name = node.name
    if name in {'script', 'style'}:
        return ''
    if name == 'table':
        return _table(node)
    if name == 'img':
        return node.get('alt', '')
    if name == 'br':
        return '\n'
    content = ''.join(_render(child) for child in node.children)
    if name in {'sup', 'sub'}:
        return ('^' if name == 'sup' else '_') + content
    if name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
        return '\n\n' + '#' * int(name[1]) + ' ' + content.strip() + '\n\n'
    if name == 'li':
        parent = node.find_parent(['ul', 'ol'])
        marker = '1. ' if parent and parent.name == 'ol' else '- '
        depth = len(node.find_parents(['ul', 'ol'])) - 1
        return '\n' + '  ' * max(0, depth) + marker + content.strip() + '\n'
    if name in {'ul', 'ol'}:
        return '\n' + content.strip('\n') + '\n'
    if name in {'p', 'div', 'section', 'blockquote', 'dl', 'figure', 'figcaption', 'hr'}:
        return '\n\n' + content.strip() + '\n\n'
    return content


def normalize_markdown(text: str) -> str:
    """Remove link targets and HTML markup without rewriting plain paragraphs."""
    protected = []

    def protect(match):
        token = f'\ufdd0{len(protected)}\ufdd1'
        if token in text:
            raise ValueError('Markdown protection token collision')
        protected.append(match.group(0))
        return token

    value = _INLINE_CODE.sub(protect, _FENCED_CODE.sub(protect, text))
    # Unknown angle-bracket notation and autolinks are visible text, not HTML.
    value = _TAG.sub(lambda m: m.group(0) if m.group(1).lower() in _HTML_TAGS else protect(m), value)
    value = _render(BeautifulSoup(value, 'html.parser'))
    parser = markdown.Markdown()
    parser.convert(value)  # Populate reference definitions, not the output text.
    value = _REFERENCE.sub(lambda m: '' if m.group(1).strip().lower() in parser.references else m.group(0), value)
    for kind in ['image_link', 'link', 'reference', 'image_reference', 'short_reference', 'short_image_ref']:
        processor = parser.inlinePatterns[kind]
        offset = 0
        while match := processor.compiled_re.search(value, offset):
            element, start, end = processor.handleMatch(match, value)
            if element is None or start is None or end is None:
                offset = match.end()
                continue
            label = element.get('alt', '') if element.tag == 'img' else (element.text or '')
            value = value[:start] + label + value[end:]
            offset = start  # Nested links/images in labels also need normalization.
    # Normalize document spacing before restoring code so literal blank lines survive.
    value = re.sub(r'\n{3,}', '\n\n', value)
    for i, original in enumerate(protected):
        value = value.replace(f'\ufdd0{i}\ufdd1', original)
    return value.strip()
