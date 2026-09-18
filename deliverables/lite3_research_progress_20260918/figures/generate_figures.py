#!/usr/bin/env python3
"""Generate matching SVG/PNG report figures using local fonts and Pillow only."""
from pathlib import Path
from html import escape
from math import hypot
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
INK = '#26394c'
MUTED = '#68798a'
LINE = '#718aa1'
BLUE = '#325f88'
PALE = '#f1f6fa'
PALE2 = '#f7f9fb'
EDGE = '#b7c7d5'

class Canvas:
    def __init__(self, height):
        self.w, self.h = 1600, height
        self.im = Image.new('RGB', (self.w, self.h), 'white')
        self.d = ImageDraw.Draw(self.im)
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{height}" viewBox="0 0 1600 {height}">', '<rect width="100%" height="100%" fill="white"/>']
    def rect(self, x, y, w, h, fill='white', stroke=EDGE, radius=12, width=2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
        self.d.rounded_rectangle((x,y,x+w,y+h),radius=radius,fill=fill,outline=stroke,width=width)
    def text(self, x, y, txt, size=29, color=INK, bold=False, anchor='middle'):
        # y denotes the top of the glyph region (not the font's ascender box).
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        bbox = self.d.textbbox((0,0),txt,font=font)
        width = self.d.textlength(txt,font=font)
        px = x-width/2 if anchor=='middle' else x
        self.d.text((px,y-bbox[1]),txt,font=font,fill=color)
        svg_anchor = 'middle' if anchor=='middle' else 'start'
        self.parts.append(f'<text x="{x}" y="{y+size*0.88}" fill="{color}" font-family="Noto Sans CJK SC, Noto Sans CJK JP, sans-serif" font-size="{size}" font-weight="{500 if bold else 400}" text-anchor="{svg_anchor}">{escape(txt)}</text>')
    def lines(self, x, y, texts, size=27, gap=41, color=INK, bold=False):
        for i,t in enumerate(texts): self.text(x,y+i*gap,t,size,color,bold)
    def path(self, points, color=LINE, width=3, dashed=False, arrow=False):
        ps=' '.join(f'{x},{y}' for x,y in points)
        dash=' stroke-dasharray="9 7"' if dashed else ''
        self.parts.append(f'<polyline points="{ps}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}/>')
        if dashed:
            for (x1,y1),(x2,y2) in zip(points,points[1:]):
                length=hypot(x2-x1,y2-y1)
                for s in range(0,int(length),16):
                    a,b=s/length,min(s+9,length)/length
                    self.d.line((x1+(x2-x1)*a,y1+(y2-y1)*a,x1+(x2-x1)*b,y1+(y2-y1)*b),fill=color,width=width)
        else: self.d.line(points,fill=color,width=width,joint='curve')
        if arrow:
            x,y=points[-1]; px,py=points[-2]
            length=hypot(x-px,y-py); ux,uy=(x-px)/length,(y-py)/length
            q=[(x,y),(x-14*ux+6*uy,y-14*uy-6*ux),(x-14*ux-6*uy,y-14*uy+6*ux)]
            self.d.polygon(q,fill=color)
            self.parts.append('<polygon points="'+' '.join(f'{a:.2f},{b:.2f}' for a,b in q)+f'" fill="{color}"/>')
    def save(self, stem):
        self.parts.append('</svg>')
        (OUT/f'{stem}.svg').write_text('\n'.join(self.parts),encoding='utf-8')
        self.im.save(OUT/f'{stem}.png',dpi=(254,254))


def architecture():
    c=Canvas(1080)
    c.text(50,22,'双候选在线学习闭环',37,BLUE,True,'start')
    c.rect(50,88,1500,100,PALE,EDGE)
    c.text(800,105,'共同起点：唯一平地 PPO 基座 W0',32,BLUE,True)
    c.text(800,151,'A / B 分别运行；每个 session 独立初始化学习参数、优化器与数据',26,MUTED)

    # Action path: one candidate is active for a given session.
    c.rect(50,322,225,116,PALE2)
    c.lines(162,340,['本体观测','o(t)'],30,45,bold=True)
    c.rect(335,235,470,128,PALE,EDGE)
    c.text(570,250,'A  LoRA-PPO · 45 维',30,BLUE,True)
    c.text(570,309,'冻结 W0，仅更新低秩参数',28)
    c.rect(335,406,470,128,PALE,EDGE)
    c.text(570,421,'B  SAC · 49 维 × 25 帧',30,BLUE,True)
    c.text(570,477,'历史接口；冻结 PPO + 动作残差',26)
    c.rect(875,322,250,116,PALE2)
    c.lines(1000,340,['共同动作过滤','限幅 / 平滑'],29,45,bold=True)
    c.rect(1190,322,360,116,PALE2)
    c.lines(1370,340,['Lite3 / Isaac Lab','执行动作 a(t)'],30,45,bold=True)
    c.path([(275,380),(302,380),(302,299),(335,299)],arrow=True)
    c.path([(302,380),(302,470),(335,470)],arrow=True)
    c.path([(805,299),(839,299),(839,380),(875,380)],arrow=True)
    c.path([(805,470),(839,470),(839,380)])
    c.path([(1125,380),(1190,380)],arrow=True)
    c.text(800,557,'统一 45 维单帧对照待验收；候选分别运行，不串联',26,MUTED)
    c.path([(1370,438),(1370,601),(162,601),(162,438)],arrow=True)
    c.rect(822,579,484,43,'white','white',0,1)
    c.text(1063,584,'新观测 o(t+1) 与回报 r(t)',27,MUTED)

    # Learning bypass, explicitly separate from the closed action loop.
    c.rect(50,666,1500,258,PALE2,EDGE)
    c.text(77,688,'学习旁路：新交互驱动参数更新与版本发布',29,BLUE,True,'start')
    c.path([(162,601),(32,601),(32,820),(75,820)],dashed=True,arrow=True)
    boxes=[(75,762,210,116),(325,762,280,116),(645,762,260,116),(945,762,230,116),(1215,762,305,116)]
    for x,y,w,h in boxes:c.rect(x,y,w,h,'white')
    c.lines(180,785,['新交互','o, a, r, o′'],28,44)
    c.lines(465,785,['PPO rollout','或 SAC replay'],28,44)
    c.lines(775,785,['参数优化','候选版本验证'],28,44)
    c.lines(1060,785,['验证通过','版本发布'],28,44)
    c.lines(1367,785,['控制端装载','执行候选版本'],28,44)
    for start,end in [(285,325),(605,645),(905,945),(1175,1215)]:c.path([(start,820),(end,820)],arrow=True)
    c.text(801,946,'冻结评测',30,BLUE,True)
    c.text(800,995,'关闭参数更新，保留已学 LoRA / 残差参数，执行相同控制链路',29)
    c.save('fig01_online_learning_architecture')


def protocol():
    c=Canvas(965)
    c.text(50,22,'独立 session 的学习与评测协议',37,BLUE,True,'start')
    c.text(800,98,'每个「地形 × 算法 × 随机种子」均从同一起点重新开始',30,MUTED)
    xs=[50,440,830,1220]
    for i,x in enumerate(xs):
        c.rect(x,181,330,206,PALE if i<2 else PALE2)
        c.text(x+30,199,f'0{i+1}',26,BLUE,True,'start')
    c.text(215,241,'干净初始化',33,BLUE,True)
    c.lines(215,294,['唯一平地 PPO','重置学习参数与数据'],26,40)
    c.text(605,241,'目标地形学习',33,BLUE,True)
    c.lines(605,294,['在线交互与更新','执行预设学习预算'],26,40)
    c.text(995,241,'冻结评测',33,BLUE,True)
    c.lines(995,294,['关闭更新，保留参数','评测目标地形表现'],26,40)
    c.text(1385,241,'平地保持检查',33,BLUE,True)
    c.lines(1385,294,['返回平地执行','检查基础能力保持'],26,40)
    for start,end in [(380,440),(770,830),(1160,1220)]:c.path([(start,284),(end,284)],arrow=True)

    c.text(800,439,'先完成开发，再以固定协议开展正式封存实验',30,BLUE,True)
    c.rect(50,508,720,245,PALE2)
    c.rect(830,508,720,245,PALE2)
    c.text(83,534,'开发阶段',32,BLUE,True,'start')
    c.text(863,534,'正式封存阶段',32,BLUE,True,'start')
    c.lines(410,602,['调试流程、超参数与预算','允许修订协议，并完整记录改动','仅作为 development 证据'],28,43)
    c.lines(1190,602,['预先固定预算、阈值、种子与判据','按同一协议重复执行独立 session','封存配置、日志、版本与评测结果'],28,43)
    c.path([(770,630),(830,630)],arrow=True)
    c.rect(50,809,1500,106,'white',EDGE)
    c.text(800,829,'旧 v10 → v24：持续迁移与开发历史',30,MUTED,True)
    c.text(800,874,'用于说明工程演进；不作为独立 session 的干净初始化来源',27,MUTED)
    c.save('fig02_independent_session_protocol')

if __name__ == '__main__':
    architecture()
    protocol()
    for p in sorted(OUT.glob('fig0*')):print(p)
