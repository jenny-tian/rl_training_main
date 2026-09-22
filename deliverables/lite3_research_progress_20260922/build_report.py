"""Build editable Word and PDF from Markdown using Pandoc and LibreOffice.

Usage: python3 build_report.py [--no-pdf]
No network access or project training dependencies are required.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

ROOT = Path(__file__).resolve().parent
STEM = 'Lite3_研究进展报告_20260922'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS = {'w': W}

def el(name, **attrs):
    return E.Element('{'+W+'}'+name, {'{'+W+'}'+k: str(v) for k,v in attrs.items()})

def child(parent, name, **attrs):
    old = parent.find('w:'+name, NS)
    if old is not None:
        parent.remove(old)
    value = el(name, **attrs)
    parent.append(value)
    return value

def ensure(parent, name):
    found = parent.find('w:'+name, NS)
    return found if found is not None else E.SubElement(parent, '{'+W+'}'+name)

def xml(root):
    return E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)

def style_docx(path):
    with ZipFile(path) as z:
        package = {i.filename: z.read(i.filename) for i in z.infolist()}
    styles = E.fromstring(package['word/styles.xml'])
    defaults = ensure(styles, 'docDefaults')
    rp = ensure(ensure(defaults, 'rPrDefault'), 'rPr')
    child(rp, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', eastAsia='Noto Serif CJK SC')
    child(rp, 'sz', val='22')
    child(rp, 'lang', val='en-US', eastAsia='zh-CN')
    pp = ensure(ensure(defaults, 'pPrDefault'), 'pPr')
    child(pp, 'spacing', before=0, after=80, line=300, lineRule='auto')
    for s in styles.findall('w:style', NS):
        sid = s.get('{'+W+'}styleId')
        if s.get('{'+W+'}type') != 'paragraph':
            continue
        p, r = ensure(s, 'pPr'), ensure(s, 'rPr')
        child(r, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', eastAsia='Noto Serif CJK SC')
        child(r, 'color', val='000000')
        if sid in ['Normal', 'BodyText', 'FirstParagraph']:
            child(r, 'sz', val=22)
            child(p, 'jc', val='both')
            child(p, 'ind', firstLine=440)
            child(p, 'spacing', before=0, after=70, line=300, lineRule='auto')
        if sid in ['Title', 'Subtitle']:
            child(p, 'ind', firstLine=0)
            child(p, 'jc', val='center')
            child(p, 'spacing', before=60, after=160 if sid=='Title' else 280, line=280, lineRule='auto')
            child(r, 'sz', val=34 if sid=='Title' else 24)
            child(r, 'b', val=1 if sid=='Title' else 0)
            n=p.find('w:numPr', NS)
            if n is not None: p.remove(n)
        if sid in ['Heading1', 'Heading2', 'Heading3']:
            level=int(sid[-1])
            child(p, 'ind', firstLine=0)
            child(p, 'spacing', before=240 if level==1 else 150, after=110, line=280, lineRule='auto')
            child(r, 'sz', val={1:29,2:25,3:22}[level])
            child(r, 'b', val=1)
            child(p, 'keepNext')
        if sid in ['Caption','ImageCaption','TableCaption']:
            child(p, 'ind', firstLine=0)
            child(p, 'jc', val='both')
            child(p, 'spacing', before=80, after=140, line=260, lineRule='auto')
            child(r, 'sz', val=19)
            child(r, 'i', val=0)
        if sid=='Compact':
            child(p, 'ind', firstLine=0)
            child(p, 'spacing', before=0, after=60, line=270, lineRule='auto')
    package['word/styles.xml']=xml(styles)
    doc=E.fromstring(package['word/document.xml'])
    sect=doc.find('.//w:sectPr',NS)
    child(sect, 'pgSz', w=11906,h=16838)
    child(sect, 'pgMar', top=1250,right=1380,bottom=1200,left=1380,header=500,footer=600,gutter=0)
    # Keep explicit page breaks compact; leave widow control on for ordinary text.
    for p in doc.findall('.//w:p',NS):
        ppr=ensure(p,'pPr')
        if p.find('.//w:br[@w:type="page"]',NS) is not None:
            child(ppr,'spacing',before=0,after=0,line=20,lineRule='exact')
            child(ppr,'ind',firstLine=0)
        if p.find('.//w:drawing',NS) is not None:
            child(ppr,'jc',val='center')
            child(ppr,'ind',firstLine=0)
            child(ppr,'keepNext')
        if p.find('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMathPara') is not None:
            child(ppr,'ind',firstLine=0)
            child(ppr,'spacing',before=80,after=100,line=260,lineRule='auto')
        text=''.join(p.itertext())
        if text.startswith('表 '):
            child(ppr,'ind',firstLine=0)
            child(ppr,'keepNext')
    table_widths = [
        [1900, 3400, 3846], [2000, 3000, 4146], [1800, 3500, 3846],
        [1650, 1450, 1800, 1500, 2746], [1650, 1500, 1800, 1500, 1100, 1596],
        [2250, 3350, 3546], [2200, 2600, 4346], [550, 1950, 6646],
    ]
    for table_index, tbl in enumerate(doc.findall('.//w:tbl',NS)):
        prop=ensure(tbl,'tblPr')
        child(prop,'tblW',w=9146,type='dxa')
        child(prop,'tblLayout',type='fixed')
        widths=table_widths[table_index]
        grid=ensure(tbl,'tblGrid')
        for gc in list(grid): grid.remove(gc)
        for width in widths: grid.append(el('gridCol',w=width))
        borders=child(prop,'tblBorders')
        for edge in ['top','bottom']:
            borders.append(el(edge,val='single',sz=8,color='607381'))
        for edge in ['left','right','insideV']:
            borders.append(el(edge,val='nil'))
        borders.append(el('insideH',val='single',sz=3,color='D4DCE1'))
        mar=child(prop,'tblCellMar')
        for edge in ['top','bottom']: mar.append(el(edge,w=80,type='dxa'))
        for edge in ['left','right']: mar.append(el(edge,w=90,type='dxa'))
        for i,tr in enumerate(tbl.findall('w:tr',NS)):
            trpr=ensure(tr,'trPr')
            child(trpr,'cantSplit')
            if i==0: child(trpr,'tblHeader')
            for col,tc in enumerate(tr.findall('w:tc',NS)):
                child(ensure(tc,'tcPr'),'tcW',w=widths[col],type='dxa')
                if i==0: child(ensure(tc,'tcPr'),'shd',fill='EEF2F5')
                for p in tc.findall('.//w:p',NS):
                    pp=ensure(p,'pPr')
                    child(pp,'ind',firstLine=0)
                    child(pp,'spacing',before=0,after=25,line=250,lineRule='auto')
                    for run in p.findall('w:r',NS):
                        rr=ensure(run,'rPr')
                        child(rr,'sz',val=18)
                        if i==0: child(rr,'b')
    # A page-number footer, matching the reference's centered page numbers.
    rels=E.fromstring(package['word/_rels/document.xml.rels'])
    rid='rIdReportFooter'
    E.SubElement(rels,'{http://schemas.openxmlformats.org/package/2006/relationships}Relationship',
                 Id=rid,Type=R+'/footer',Target='footer_report.xml')
    fr=E.Element('{'+W+'}ftr',nsmap={'w':W})
    p=E.SubElement(fr,'{'+W+'}p')
    child(ensure(p,'pPr'),'jc',val='center')
    fld=E.SubElement(p,'{'+W+'}fldSimple',{'{'+W+'}instr':' PAGE '})
    r=E.SubElement(fld,'{'+W+'}r'); child(ensure(r,'rPr'),'sz',val=20)
    E.SubElement(r,'{'+W+'}t').text='1'
    fref=E.Element('{'+W+'}footerReference',{'{'+W+'}type':'default','{'+R+'}id':rid})
    sect.insert(0,fref)
    package['word/footer_report.xml']=xml(fr)
    package['word/_rels/document.xml.rels']=xml(rels)
    ct=E.fromstring(package['[Content_Types].xml'])
    E.SubElement(ct,'{http://schemas.openxmlformats.org/package/2006/content-types}Override',
                 PartName='/word/footer_report.xml',ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml')
    package['[Content_Types].xml']=xml(ct)
    package['word/document.xml']=xml(doc)
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for k,v in package.items(): z.writestr(k,v)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--no-pdf',action='store_true'); args=parser.parse_args()
    pandoc=shutil.which('pandoc') or '/home/tjx/anaconda3/bin/pandoc'
    out=ROOT/(STEM+'.docx')
    subprocess.run([pandoc,'report.md','--from=markdown+tex_math_dollars+raw_attribute',
                    '--resource-path=.', '--output='+str(out)],cwd=ROOT,check=True)
    style_docx(out)
    if not args.no_pdf:
        with tempfile.TemporaryDirectory(prefix='lite3_lo_') as profile:
            subprocess.run(['libreoffice','-env:UserInstallation=file://'+profile,'--headless',
                            '--convert-to','pdf:writer_pdf_Export','--outdir',str(ROOT),str(out)],check=True,timeout=120)
    print(out)

if __name__=='__main__': main()
