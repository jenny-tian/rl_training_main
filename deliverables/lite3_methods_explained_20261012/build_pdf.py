"""Build the Chinese methods guide with a TOC, equations and vector diagrams."""
from pathlib import Path
import html
import json
import re
import subprocess

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, PageBreak, Table, TableStyle, Image, Flowable, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'Lite3_仓库方法原理详解.md'
PDF = HERE / 'Lite3_仓库方法原理详解_20261012.pdf'
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
MATH_PYTHON = '/home/tjx/anaconda3/envs/env_isaaclab/bin/python'

FORMULAS = re.findall(r'^\$\$ (.+?) \$\$$', SOURCE.read_text(), re.M)
formula_json = ASSETS / 'formulas.json'
formula_json.write_text(json.dumps(FORMULAS, ensure_ascii=False, indent=2))
render_code = r'''
import json, sys, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['mathtext.fontset']='stix'
p=Path(sys.argv[1])
for i,eq in enumerate(json.loads(p.read_text())):
 eq=re.sub(r'(\\(?:mathcal|mathbb|widehat))\s+([A-Za-z])',r'\1{\2}',eq)
 eq=re.sub(r'\\le(?![A-Za-z])',r'\\leq ',eq)
 eq=re.sub(r'\\ge(?![A-Za-z])',r'\\geq ',eq)
 fig=plt.figure(figsize=(13,1),dpi=180)
 fig.text(.01,.5,'$'+eq+'$',fontsize=17,va='center',color='#17334b')
 fig.savefig(p.parent/f'equation-{i:02d}.png',bbox_inches='tight',pad_inches=.10,transparent=True)
 plt.close(fig)
'''
subprocess.run([MATH_PYTHON, '-c', render_code, str(formula_json)], check=True)

pdfmetrics.registerFont(TTFont('CN', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
pdfmetrics.registerFont(TTFont('Latin', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('LatinBold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CN', italic='CN', boldItalic='CN')
CN_GLYPHS = pdfmetrics.getFont('CN').face.charToGlyph
REPO_ROOT = HERE.parents[1]
BASE_URL = 'https://github.com/jenny-tian/rl_training_main/blob/62a58992712cf6132da38eb4f606c14c25d063b5/'
BLUE = colors.HexColor('#153b57')
TEAL = colors.HexColor('#137e8c')
LIGHT = colors.HexColor('#eef5f8')
GRAY = colors.HexColor('#5a6b78')
BODY_WIDTH = A4[0] - 100
styles = {
 'body': ParagraphStyle('body',fontName='CN',fontSize=10.2,leading=17.0,spaceAfter=8,wordWrap='CJK',textColor=colors.HexColor('#263746')),
 'h1': ParagraphStyle('h1',fontName='CN',fontSize=18.4,leading=26,spaceAfter=15,textColor=BLUE,wordWrap='CJK',keepWithNext=True),
 'h2': ParagraphStyle('h2',fontName='CN',fontSize=12.4,leading=19,spaceBefore=10,spaceAfter=7,textColor=TEAL,wordWrap='CJK',keepWithNext=True),
 'table': ParagraphStyle('table',fontName='CN',fontSize=8.5,leading=13.5,wordWrap='CJK',textColor=BLUE),
 'small': ParagraphStyle('small',fontName='CN',fontSize=8.6,leading=14.5,spaceAfter=7,wordWrap='CJK',textColor=GRAY),
 'cover': ParagraphStyle('cover',fontName='CN',fontSize=31,leading=43,textColor=BLUE,wordWrap='CJK'),
 'subtitle': ParagraphStyle('subtitle',fontName='CN',fontSize=15,leading=25,textColor=TEAL,spaceAfter=14,wordWrap='CJK'),
}

def mixed(text):
    chunks = re.split(r'(<[^>]+>)', text)
    result=[]
    for chunk in chunks:
        if chunk.startswith('<'):
            result.append(chunk);continue
        raw=html.unescape(chunk)
        runs=[];last=None;run=''
        for char in raw:
            font='CN' if ord(char) in CN_GLYPHS else 'Latin'
            if font!=last and run:
                runs.append(f'<font name="{last}">{html.escape(run)}</font>');run=''
            run+=char;last=font
        if run:runs.append(f'<font name="{last}">{html.escape(run)}</font>')
        result.extend(runs)
    return ''.join(result)

def inline(text):
    text = html.escape(text)
    def code(m):
        value = m.group(1)
        # Optional invisible break points keep long source paths within margins.
        value = value.replace('/', '/\u200b').replace('_', '_\u200b')
        label='<font color="#42617a" size="9">'+value+'</font>'
        path=html.unescape(m.group(1))
        if (path.startswith(('docs/','source/','scripts/')) and (REPO_ROOT/path).exists()):
            label=f'<link href="{BASE_URL+path}">{label}</link>'
        return label
    text = re.sub(r'`([^`]+)`', code, text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return mixed(text)

class Diagram(Flowable):
    def __init__(self, kind):
        Flowable.__init__(self)
        self.kind = kind
        self.width = BODY_WIDTH
        self.height = 165 if kind in ('sac','flow','recovery') else 145
    def wrap(self, w, h):
        return self.width, self.height
    def draw(self):
        c = self.canv
        def box(x,y,w,h,title,detail='',fill=LIGHT):
            c.setFillColor(fill);c.setStrokeColor(colors.HexColor('#c9dce5'))
            c.roundRect(x,y,w,h,5,fill=1,stroke=1)
            labelstyle=ParagraphStyle('diagramlabel',fontName='CN',fontSize=9.7,leading=12,alignment=TA_CENTER,textColor=BLUE)
            p=Paragraph(mixed(html.escape(title)),labelstyle);pw,ph=p.wrap(w-8,h);p.drawOn(c,x+4,y+h-10-ph)
            if detail:
                c.setFont('CN',8.2);c.setFillColor(GRAY)
                for i,line in enumerate(detail.split('|')):
                    labelstyle=ParagraphStyle('diagramdetail',fontName='CN',fontSize=8.0,leading=10,alignment=TA_CENTER,textColor=GRAY)
                    p=Paragraph(mixed(html.escape(line)),labelstyle);pw,ph=p.wrap(w-8,h);p.drawOn(c,x+4,y+h-27-i*12-ph)
        def arrow(x1,y1,x2,y2):
            c.setStrokeColor(TEAL);c.setFillColor(TEAL);c.setLineWidth(1.1)
            c.line(x1,y1,x2,y2)
            import math
            a=math.atan2(y2-y1,x2-x1)
            p=c.beginPath();p.moveTo(x2,y2)
            p.lineTo(x2-6*math.cos(a-.45),y2-6*math.sin(a-.45))
            p.lineTo(x2-6*math.cos(a+.45),y2-6*math.sin(a+.45));p.close()
            c.drawPath(p,fill=1,stroke=0)
        w=(self.width-28)/3
        if self.kind=='overview':
            for i,(t,d) in enumerate([('已有平地策略','基本支撑与关节协调'),('崎岖行走适应','完整PPO / SAC / LoRA'),('失稳恢复开发','恢复教师 / 搜索 / 迁移')]):
                box(i*(w+14),70,w,58,t,d)
                if i<2:arrow(i*(w+14)+w,99,(i+1)*(w+14),99)
            c.setFont('CN',9);c.setFillColor(GRAY)
            p=Paragraph(mixed('FlowDAgger：在冻结生成先验上，用纠正监督学习噪声策略'),ParagraphStyle('diagramfooter',fontName='CN',fontSize=9,leading=12,alignment=TA_CENTER,textColor=GRAY));p.wrap(self.width,30);p.drawOn(c,0,44)
            c.drawCentredString(self.width/2,25,'每条路线都需要冻结评测；恢复教师资格是新集成的前置条件')
        elif self.kind=='control':
            items=[('读取观测','机身 / 关节 / 路线'),('策略与动作接口','网络输出 → 限幅'),('物理执行','目标位置 → PD → 接触')]
            for i,(t,d) in enumerate(items):
                box(i*(w+14),70,w,58,t,d)
                if i<2:arrow(i*(w+14)+w,99,(i+1)*(w+14),99)
            arrow(self.width-25,70,self.width-25,37);arrow(self.width-25,37,24,37);arrow(24,37,24,70)
            c.setFillColor(GRAY);c.setFont('CN',9)
            p=Paragraph(mixed('下一状态、奖励与终止 → 更新观测；训练时再用于学习'),ParagraphStyle('controlfooter',fontName='CN',fontSize=9,leading=12,alignment=TA_CENTER,textColor=GRAY));p.wrap(self.width,30);p.drawOn(c,0,43)
        elif self.kind=='sac':
            box(0,91,w,60,'冻结基础PPO','产生基础动作')
            box(w+14,91,w,60,'GRU + 残差Actor','历史 → 有界增量')
            box(2*(w+14),91,w,60,'安全过滤与物理','基础 + 增量 → 执行')
            arrow(w-20,151,w-20,159);arrow(w-20,159,self.width-20,159);arrow(self.width-20,159,self.width-20,151)
            arrow(2*w+14,120,2*(w+14),120)
            box(0,8,w,60,'经验回放','请求 / 执行 / 奖励')
            box(w+14,8,w,60,'双Q与目标网络','估计软价值')
            box(2*(w+14),8,w,60,'策略更新与发布','更新残差Actor')
            arrow(self.width-20,91,20,68);arrow(w,38,w+14,38);arrow(2*w+14,38,2*(w+14),38)
            arrow(self.width-20,68,2*w+4,91)
        elif self.kind=='flow':
            for i,(t,d) in enumerate([('学生访问状态','噪声网络 → 冻结flow'),('教师纠正采集','请求动作 + 配对收益'),('离散反演与验收','重建误差 / 噪声边界')]):
                box(i*(w+14),91,w,60,t,d)
                if i<2:arrow(i*(w+14)+w,120,(i+1)*(w+14),120)
            box(0,8,w,60,'纠正 / 自主双缓冲','严格50% / 50%')
            box(w+14,8,w,60,'监督噪声网络','flow参数不更新')
            box(2*(w+14),8,w,60,'冻结学生评测','推理时没有教师接管')
            arrow(self.width-20,91,20,68);arrow(w,38,w+14,38);arrow(2*w+14,38,2*(w+14),38)
        else:
            for i,(t,d) in enumerate([('困难TRAIN来源','早期失败 / 接触诊断'),('36维短动作搜索','3段 × 4步 × 12关节'),('延迟筛选与跟进','找到可完成的固定前缀')]):
                box(i*(w+14),91,w,60,t,d)
                if i<2:arrow(i*(w+14)+w,120,(i+1)*(w+14),120)
            box(0,8,w,60,'新采集成功示范','观测 → 执行动作')
            box(w+14,8,w,60,'完整Actor监督迁移','恢复 + 普通行为保持')
            box(2*(w+14),8,w,60,'PPO后段优化','路线精度 / 资格 / 复现')
            arrow(self.width-20,91,20,68);arrow(w,38,w+14,38);arrow(2*w+14,38,2*(w+14),38)

class Guide(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=57,bottomMargin=53,
            title='Lite3 仓库方法原理详解',author='Lite3 项目方法说明',pageCompression=1)
        self.addPageTemplates(PageTemplate(id='normal',frames=[Frame(50,53,BODY_WIDTH,A4[1]-110,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.header_footer))
    def header_footer(self,c,doc):
        c.saveState()
        if doc.page>1:
            hstyle=ParagraphStyle('header',fontName='CN',fontSize=8,leading=10,textColor=GRAY)
            p=Paragraph(mixed('Lite3 仓库方法原理详解'),hstyle);p.wrap(260,20);p.drawOn(c,50,A4[1]-32)
            p=Paragraph(mixed('资料截止 2026-10-12'),hstyle);p.wrap(110,20);p.drawOn(c,A4[0]-160,A4[1]-32)
            c.setStrokeColor(colors.HexColor('#d8e4eb'));c.line(50,A4[1]-40,A4[0]-50,A4[1]-40)
        c.setFont('Latin',8);c.setFillColor(GRAY)
        c.drawCentredString(A4[0]/2,29,str(doc.page))
        c.restoreState()
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name=='h1' and hasattr(f,'section_id'):
            key=f.section_id
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(f.getPlainText(),key,0,False)
            self.notify('TOCEntry',(0,mixed(html.escape(f.getPlainText())),self.page,key))

def build_story():
    story=[Spacer(1,92),Paragraph(mixed('Lite3<br/>仓库方法原理详解'),styles['cover']),Spacer(1,24),
        Paragraph(mixed('从观测与关节控制，到 PPO、在线适应、<br/>FlowDAgger 与失稳恢复'),styles['subtitle']),
        Spacer(1,28),Diagram('overview'),Spacer(1,20),
        Paragraph('结合实际源码、训练流程、数学公式与保存的实验结果。<br/>每种方法均说明学习对象、训练信号、部署流程及已验证范围。',styles['body']),
        Spacer(1,19),Paragraph(mixed('资料截止：2026年10月12日<br/>技术基线：62a5899　｜　中文详细说明'),styles['small']),PageBreak(),
        Paragraph('目录与阅读建议',styles['h1']),
        Paragraph(mixed('先读第1～6章理解共同控制与PPO基础；第7～11章比较适配路线；第12～14章是近期恢复开发；第16～20章说明评测、限制和源码入口。'),styles['body'])]
    toc=TableOfContents()
    toc.levelStyles=[ParagraphStyle('toc',fontName='Latin',fontSize=9.6,leading=17,spaceBefore=1,wordWrap='CJK',textColor=BLUE)]
    toc.tableStyle=TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),8)])
    story.extend([toc,PageBreak()])
    lines=SOURCE.read_text().splitlines();i=0;eq_index=0;chapter=0
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith('# ') or (i<6):i+=1;continue
        if line.startswith('## '):
            if chapter:
                story.extend([CondPageBreak(280),Spacer(1,18)])
            chapter+=1
            p=Paragraph(inline(line[3:]),styles['h1']);p.section_id=f'section-{chapter}'
            story.append(p);i+=1;continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]),styles['h2']));i+=1;continue
        if line.startswith('@diagram '):
            story.extend([Spacer(1,7),Diagram(line.split()[1]),Spacer(1,9)]);i+=1;continue
        if line.startswith('$$ '):
            p=ASSETS/f'equation-{eq_index:02d}.png';eq_index+=1
            from PIL import Image as PILImage
            iw,ih=PILImage.open(p).size
            factor=min(BODY_WIDTH/iw,0.32)
            im=Image(str(p),width=iw*factor,height=ih*factor);im.hAlign='CENTER'
            story.extend([Spacer(1,3),im,Spacer(1,11)]);i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                raw=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',v.replace(' ','')) for v in raw):
                    rows.append([Paragraph(inline(v),styles['table']) for v in raw])
                i+=1
            n=len(rows[0]);widths=[BODY_WIDTH/n]*n
            if n==4:widths=[BODY_WIDTH*x for x in (.20,.24,.25,.31)]
            if n==5:widths=[BODY_WIDTH*x for x in (.12,.10,.10,.35,.33)]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf2')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),8),
                ('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),.6,TEAL)]))
            story.extend([table,Spacer(1,12)]);continue
        if line.startswith('- '):
            story.append(Paragraph(inline('• '+line[2:]),styles['small'] if chapter==20 else styles['body']));i+=1;continue
        parts=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','$$','@diagram','- ')):
            parts.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(parts)),styles['body']))
    return story

Guide(PDF).multiBuild(build_story())
print(PDF)
print(f'{len(FORMULAS)} formulas rendered')
