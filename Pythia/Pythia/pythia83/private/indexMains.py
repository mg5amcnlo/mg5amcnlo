#!/usr/bin/env python
# Copyright (C) 2021 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

# Author: Christian Bierlich <christian.bierlich@thep.lu.se>

# This script is used to build the example index, execute as
# "./private/indexMains.py". Note, this should be run after convertXML.
import glob, re, collections, os, shutil

# Common HTML header.
header = """<html><head><title>{title}</title>
<link rel="stylesheet" type="text/css" href="../pythia.css"/>
<link rel="shortcut icon" href="../pythia32.gif"/></head><body><h2>{title}</h2>
Back to <a href="../ExampleKeywords.html" target="page">index.</a>
<pre><code class="language-{language}">
"""

# Parse the extracted information.
def parse(key, vals, info):
    vals, parsed = vals.replace(",", ";").split(";"), []
    for val in vals:
        val = val.split(">")[0].strip()
        if not val: continue

        # Authors and contacts.
        if key in ("authors", "contacts"):
            if not "<" in val: name, mail = val.replace(".", "").strip(), ""
            else: name, mail = tuple(val.split("<"))
            info[key] += [(name.strip(), mail)]
            parsed += ["<a href=\"mailto:%s\">%s</a>" % (mail, name.strip())
                       if mail else name]

        # Keywords.
        else:
            link = val.lower().replace(" ", "+")
            text = val.capitalize() if val.islower() else val
            name = text.replace(" ", "&nbsp;").replace("-", "&#8209;")
            info[key] += [(link, text, name)]
            parsed += ["<a href=\"../ExampleKeywords.html#%s\">%s</a>" %
                       (link, name)]
    return parsed

# Extract the information from an example main and write to HTML.
def extract(name, path):

    # Set the comment type and define the default file information.
    comment = "#" if ".py" in name else "//"
    info = {"title": name[9:-3].lower() + ("-python" if ".py" in name else ""),
            "name": name[9:].lower(), "authors": [], "contacts": [],
            "keywords": [], "language": "python" if ".py" in name else "c++"}

    # Open the code input and HTML output. Parse the file.
    code = file(name, "r")
    html = file(path + "/" + info["title"] + ".html", "w")
    html.write(header.format(**info))
    top, tag, now = True, None, None
    for verb in code:

        # Check for keys.
        line, blank = verb.strip(), False
        if not top: html.write(verb); continue
        elif not line: now = verb
        elif not line.startswith(comment): now = verb; top = True
        else:
            line = line.split(comment, 1)[1].strip()
            try: key, val = (item.strip() for item in line.split(":", 1))
            except: key, val = None, None
            if key and key.lower() in info:
                if tag: now = [key.lower(), val]
                else: tag = [key.lower(), val]
            elif tag: tag[1] += "; " + line
            else: now = verb

        # Write HTML and update info.
        if tag and now:
            pre = "\n" + comment + " "*12 
            html.write(comment + " " + tag[0].capitalize() + ":" + pre
                       + pre.join(parse(tag[0], tag[1], info)) + "\n")
            tag = None
        if type(now) is list: tag = now
        elif now: html.write(now)
        now = None

    # Close the file and return.
    code.close()
    html.write("</code></pre></body></html>")
    html.close()
    return info

# Create the output directory.
path = "share/Pythia8/htmldoc/examples"
if os.path.exists(path): shutil.rmtree(path)
os.makedirs(path)

# Create the description database.
xml = file("share/Pythia8/xmldoc/SampleMainPrograms.xml")
descriptions, key, val = {}, None, None
for verb in xml:
    line = "".join(verb.split())
    if line.startswith("<li><code>main") and ":" in line:
        key = line[10:].split("</code>")[0].lower()
        val = verb.split(":")[-1].strip()
    elif key: val += " " + verb.strip()
    if key and line.endswith("</li>"):
        descriptions[key], key, val = val, None, None
xml.close()

# Extract information from the examples.
keywords = {}
for name in glob.glob("examples/*.cc") + glob.glob("examples/*py"):
    info = extract(name, path)
    for key, text, name in info["keywords"]:
        if key in keywords: keywords[key][2] += [info]
        else: keywords[key] = [text, name, [info]]

# Create the keywords page header.
html = file("share/Pythia8/htmldoc/ExampleKeywords.html", "w")
html.write("""
<html><head><title>Example Keywords</title>
<link rel="stylesheet" type="text/css" href="pythia.css"/>
<link rel="shortcut icon" href="../pythia32.gif"/></head><body>
""")
xml = file("share/Pythia8/xmldoc/ExampleKeywords.xml")
for verb in xml: html.write(verb)
xml.close()
html.write("""
<button class="expand">Click to show the list of keywords.</button>
<div class="panel"><p style="line-height:2">&#9679;
""")

# Write the keyword links, full list, and search database.
pre, links, full, index = "&ensp;"*4 + "&#9679;&nbsp;", [], [], []
for key in sorted(keywords.keys(), key = lambda s: s.lower()):
    text, name, infos = keywords[key]
    full += ["<a name=%s></a><h3>%s</h3><ul>" % (key, name)]
    links += ["<a href=#%s>%s</a>" % (key, name)]
    index += [{"name": name, "link": "ExampleKeywords.html#" + key,
               "text": text}]
    for info in sorted(infos, key = lambda d: d["name"]):
        if not "description" in info:
            if info["name"] in descriptions:
                info["description"] = descriptions[info["name"]]
            else:
                info["description"] = ""
                print("%s is missing a description" % info["name"])
        full += ["<li><a href=examples/{title}.html>{name}</a>: {description}"
                 "</li>".format(**info)]
    full += ["</ul>\n"]
html.write(pre.join(links) + "</p></div>")
html.write("\n".join(full))
script = file("share/Pythia8/htmldoc/Examples.js", "w")
script.write("var index = %r" % index)
script.close()

# Write the footer.
html.write("<script src=\"Expand.js\"></script></body></html>")
