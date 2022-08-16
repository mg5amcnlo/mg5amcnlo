#!/usr/bin/env python
# Copyright (C) 2021 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, May 2017.

# This script is used to build the search index, execute as
# "./private/indexXML.py".
import glob, re, collections
import xml.etree.cElementTree as xml

# Clean the special characters in a line.
def clean(line, html = False):
    if not line: return line
    d = [('\n', ''), ('@', '\@'), ('AMPERSAND', '&'), ('"', '\'')]
    if not html: d += [('&lt;', '<'), ('&gt;', '>'), ('&lt', '<'), ('&gt', '>')]
    for f, r in d: line = line.replace(f, r)
    line = line.strip()
    return line

# Clean the special characters in a line and append to a list.
def append(text, line):
    line = clean(line)
    if line: text += [line]

# Parse an XML element and write it to the index.
def parse(index, base, element, cat = None, key = None, lnk = 0):
    if element.tag in ['particle', 'channel']: return lnk
    if 'name' in element.attrib and element.tag not in ['argument']:
        key = clean(element.attrib['name'], True)
        if element.tag not in ['chapter']:
            lnk += 1; cat = 'method' if 'method' in element.tag else 'setting'
        else: cat = 'chapter'
    if key not in index[cat]:
        index[cat][key] = {'link':'%s.html' % base, 'text': []}
        if cat != 'chapter': index[cat][key]['link'] += '#anchor%i' % lnk
    append(index[cat][key]['text'], key)
    append(index[cat][key]['text'], element.text)
    for child in element: lnk = parse(index, base, child, cat, key, lnk)
    append(index[cat][key]['text'], element.tail)
    return lnk

# The main script.
names  = glob.glob('share/Pythia8/xmldoc/*.xml')
ignore = ['Bibliography', 'Frontpage', 'Glossary', 'Index', 'ProgramClasses',
          'ProgramFiles', 'ProgramMethods', 'SaveSettings', 'UpdateHistory',
          'Version', 'Welcome']
index  = collections.OrderedDict([
        ('chapter', {}), ('setting', {}), ('method', {})])
for name in names:
    xmldoc = file(name).read().replace('&', 'AMPERSAND')
    name = name.split('/')[-1].replace('.xml', '')
    if name in ignore: continue
    tree = xml.fromstring(xmldoc)
    parse(index, name, tree)
out, idx = file('share/Pythia8/htmldoc/Index.js', 'w'), 0
out.write('var index = [')
for cat, dct in index.iteritems():
    for key, val in sorted(dct.iteritems(), key = lambda s: s[0].lower()):
        out.write('%s{"name":"%s","link":"%s","text":"%s"}' % (
                (',' if idx else ''), key, val['link'],
                re.sub(r'\s([ ?.!"](?:\s|$))', r'\1', ' '.join(val['text']))))
        idx += 1
out.write('];')
out.close()
