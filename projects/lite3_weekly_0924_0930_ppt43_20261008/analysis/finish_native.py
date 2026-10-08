from pathlib import Path
import hashlib
import json
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement

P = Path(__file__).resolve().parents[1]
source = max((P/'analysis').glob('*工作汇报*.pptx'),key=lambda p:p.stat().st_mtime)
prs = Presentation(source)
assert len(prs.slides)==4
video = (P/'media/teacher_0923_recorded_0924_zoom.mp4').read_bytes()
poster = (P/'media/teacher_poster.jpg').read_bytes()
media_count=0
for slide in prs.slides:
    for shape in slide.shapes:
        if shape.shape_type == 16:
            media_count+=1
            for rel in slide.part.rels.values():
                if rel.reltype.endswith('/video') or rel.reltype.endswith('/media'):
                    assert not rel.is_external
                    rel.target_part._blob=video
            blip = shape._element.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
            rid=blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            slide.part.rels[rid].target_part._blob=poster
            nv=shape._element.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr')
            nv.set('name','旧教师模型_9月24日补录_扫描修复前')
            nv.set('descr','旧版教师在第683个控制步跌倒；不代表扫描修复后的行走效果。')
        if shape.has_chart:
            chart=shape.chart
            chart.value_axis.minimum_scale=0
            chart.value_axis.maximum_scale=9
            chart.value_axis.major_unit=2
            chart.value_axis.tick_labels.number_format='0'
            chart.value_axis.tick_labels.font.size=Pt(10)
            chart.category_axis.tick_labels.font.size=Pt(11)
            for s in chart.series:
                s.data_labels.number_format='0.00'
                s.data_labels.number_format_is_linked=False
                s.data_labels.font.size=Pt(13)
                for i,pt in enumerate(s.points):
                    pt.format.fill.solid()
                    pt.format.fill.fore_color.rgb=RGBColor.from_string('004098' if i%2==0 else 'C57B35')
                    pt.format.line.fill.background()
            shape.height=int(181*9525)
            # Keep four category labels legible in the original chart slot.
            for p in chart.part._element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}bodyPr'):
                p.set('rot','0')

assert media_count==0

def set_font(run,size,color,bold=False,name='宋体'):
    run.font.name=name;run.font.size=Pt(size)
    run.font.color.rgb=RGBColor.from_string(color);run.font.bold=bold
    ea=OxmlElement('a:ea');ea.set('typeface',name);run._r.get_or_add_rPr().append(ea)

# The native text filler copies the first run's styling. Restore the template's
# blue lead-in and black body instead of making entire paragraphs blue and bold.
for index,ids in [(1,{122,123,124,125}),(2,{115}),(3,{118,119,120})]:
    for shape in prs.slides[index].shapes:
        if shape.shape_id not in ids or not shape.has_text_frame:continue
        text=shape.text_frame.text
        tf=shape.text_frame;tf.clear();tf.word_wrap=True
        p=tf.paragraphs[0]
        prefix,sep,rest=text.partition('：')
        if sep:
            r=p.add_run();r.text=prefix+sep;set_font(r,17.5,'004098',True)
            r=p.add_run();r.text=rest;set_font(r,17.5,'222222')
        else:
            r=p.add_run();r.text=text;set_font(r,17.5,'222222')

cover=next(sh for sh in prs.slides[0].shapes if sh.has_text_frame)
cover.top=480*9525;cover.height=136*9525
tf=cover.text_frame;tf.clear();tf.vertical_anchor=MSO_ANCHOR.MIDDLE
for i,text in enumerate(['机器人组 田家鑫组会汇报','2026.9.24—9.30']):
    p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
    p.alignment=PP_ALIGN.CENTER;p.space_after=Pt(8)
    r=p.add_run();r.text=text;set_font(r,28 if i==0 else 24,'FFFFFF',True,'黑体')

def caption(slide,text,y=616):
    tb=slide.shapes.add_textbox(62*9525,y*9525,835*9525,44*9525)
    tf=tb.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=0
    r=tf.paragraphs[0].add_run();r.text=text
    r.font.name='宋体';r.font.size=Pt(13);r.font.color.rgb=RGBColor.from_string('666666')
    ea=OxmlElement('a:ea');ea.set('typeface','宋体');r._r.get_or_add_rPr().append(ea)

caption(prs.slides[2],'9/29 独立确认：每地形 2 个新回合；模型与参数提前固定，未按结果追加调参。')

out=P/'exports'/source.name
prs.core_properties.title='田家鑫组会工作汇报：2026.9.24—9.30'
prs.core_properties.subject='扫描修复、基线统一、SAC修复、价值网络与策略更新对照'
prs.core_properties.author='田家鑫'
prs.core_properties.keywords='Lite3;2026-09-24;2026-09-30;工作汇报'
prs.save(out)
(P/'analysis/native_finish.json').write_text(json.dumps({
    'source':source.name,'output':out.name,'slide_count':4,
    'media_count':0,'video_included':False,
    'edits':['数据图轴从控制步改为米，范围0—9','参考与高额度柱分别蓝色和棕色','新增实验范围说明'],
    'video_addresses_on_slides':False},ensure_ascii=False,indent=2))
print(out)
