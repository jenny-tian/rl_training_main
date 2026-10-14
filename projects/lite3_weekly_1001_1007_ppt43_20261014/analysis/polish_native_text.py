"""Restore template lead/body styling after the native text fill.

The template writer copies the first run's style for replacement text. These
slots originally use a blue bold lead and black regular body. Restore that
pattern without changing layout, fonts, text, tables, or any other parts.
"""
from pathlib import Path
from copy import deepcopy
import json
import zipfile
import lxml.etree as E

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=next((PROJECT/'sources').glob('*.pptx'))
OUTPUT=next((PROJECT/'exports').glob('*.pptx'))
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'rel':'http://schemas.openxmlformats.org/package/2006/relationships'}
A='{'+NS['a']+'}'

def props(root,shape_id):
    shape=root.xpath(f'.//p:sp[p:nvSpPr/p:cNvPr[@id="{shape_id}"]]',namespaces=NS)[0]
    runs=shape.findall('.//a:p/a:r',NS)
    return deepcopy(runs[0].find('a:rPr',NS)),deepcopy(runs[1].find('a:rPr',NS))

with zipfile.ZipFile(SOURCE) as z:
    source_roots={i:E.fromstring(z.read(f'ppt/slides/slide{i}.xml')) for i in [2,3]}
    styles={i:props(source_roots[2 if i in [122,123,124,125] else 3],i) for i in [122,123,124,125,115]}

with zipfile.ZipFile(OUTPUT) as z:
    entries={x.filename:(x,z.read(x.filename)) for x in z.infolist()}
presentation=E.fromstring(entries['ppt/presentation.xml'][1])
rels=E.fromstring(entries['ppt/_rels/presentation.xml.rels'][1])
import posixpath
targets={r.get('Id'):posixpath.normpath(posixpath.join('ppt',r.get('Target'))) for r in rels}
changed=[]
for page,sid in enumerate(presentation.findall('p:sldIdLst/p:sldId',NS),1):
    part=targets[sid.get('{'+NS['r']+'}id')]
    root=E.fromstring(entries[part][1])
    ids=[122,123,124,125] if page==2 else [115] if page in [3,4] else []
    for shape_id in ids:
        shape=root.xpath(f'.//p:sp[p:nvSpPr/p:cNvPr[@id="{shape_id}"]]',namespaces=NS)[0]
        lead,body=styles[shape_id]
        for paragraph in shape.findall('p:txBody/a:p',NS):
            text=''.join(paragraph.xpath('.//a:t/text()',namespaces=NS))
            if not text:continue
            cut=text.find('：')+1
            if cut==0:cut=0
            chunks=[(text[:cut],lead),(text[cut:],body)]
            for child in list(paragraph):
                if child.tag not in [A+'pPr',A+'endParaRPr']:paragraph.remove(child)
            end=paragraph.find('a:endParaRPr',NS)
            for value,style in chunks:
                if not value:continue
                run=E.Element(A+'r');run.append(deepcopy(style));E.SubElement(run,A+'t').text=value
                if end is None:paragraph.append(run)
                else:paragraph.insert(paragraph.index(end),run)
        changed.append({'slide':page,'shape_id':shape_id})
    if ids:entries[part]=(entries[part][0],E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True))
temporary=PROJECT/'analysis/polished.tmp.pptx'
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for info,data in entries.values():z.writestr(info,data)
temporary.replace(OUTPUT)
(PROJECT/'analysis/style_polish.json').write_text(json.dumps({'scope':'Restore source lead/body run formatting only','changed':changed},ensure_ascii=False,indent=2))
print('Restored native lead/body styling:',len(changed),'text frames')
