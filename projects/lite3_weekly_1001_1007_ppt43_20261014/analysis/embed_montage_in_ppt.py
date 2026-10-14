"""Append an explicitly requested native video page, preserving existing pages."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import zipfile

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'exports/田家鑫_10.1-10.7_组会汇报_4页_20261014_091453.pptx'
VIDEO=PROJECT/'media/完整PPO微调_六段拼接对比_2倍速_20261014.mp4'
POSTER=PROJECT/'media/完整PPO微调_演示封面.png'


def replace_text_preserving_style(shape,text):
    first=shape.text_frame.paragraphs[0]
    style=deepcopy(first.runs[0]._r.rPr) if first.runs else None
    shape.text_frame.clear()
    paragraph=shape.text_frame.paragraphs[0]
    run=paragraph.add_run();run.text=text
    if style is not None:
        old=run._r.rPr
        if old is not None:run._r.remove(old)
        run._r.insert(0,style)


def main():
    prs=Presentation(SOURCE)
    assert len(prs.slides)==4 and VIDEO.exists() and POSTER.exists()
    source=prs.slides[2]
    page=prs.slides.add_slide(source.slide_layout)
    for shape in list(page.shapes):
        el=shape._element;el.getparent().remove(el)
    # Title, page number and footer carry the exact original template geometry.
    for shape in source.shapes:
        if shape.shape_id in (112,114,117):
            page.shapes._spTree.insert_element_before(deepcopy(shape.element),'p:extLst')
    for shape in page.shapes:
        if shape.shape_id==112:replace_text_preserving_style(shape,'完整PPO微调：多地形视频对比')
        elif shape.shape_id==114:replace_text_preserving_style(shape,'3')
        elif shape.shape_id==117:
            replace_text_preserving_style(shape,'10/14 新补录；与原59/60验收分开。原速分段视频一并保留。')
            shape.top=Inches(6.55)
    caption=page.shapes.add_textbox(Inches(.65),Inches(1.78),Inches(8.7),Inches(.30))
    p=caption.text_frame.paragraphs[0];r=p.add_run()
    r.text='六段拼接 · 3个模型 × 2种地形 · 点击画面播放（2倍速）'
    r.font.name='Microsoft YaHei';r.font.size=Pt(15);r.font.color.rgb=RGBColor(85,85,85)
    movie=page.shapes.add_movie(str(VIDEO),Inches(.65),Inches(2.12),Inches(8.7),Inches(4.35),
        poster_frame_image=str(POSTER),mime_type='video/mp4')
    movie.name='完整PPO微调_六段拼接演示_点击播放'
    page.notes_slide.notes_text_frame.text=(
        '这一页是十月十四日新补录的演示，不是原六十次验收的视频。先看同一起点下原平地模型与微调模型二百八十三在两种崎岖地形的对比，再看独立训练模型三百零七的两段行走。'
        '六段视频全部使用提前固定的首回合，没有根据结果换起点；拼接画面以两倍速播放，较短片段结束后停留在最后一帧。'
        '底部显示本次单回合的横向偏移均方根与峰值，它们不能代替原六地形统计或证明任意场景的稳定性。')
    # Move the new video page before the original result table.
    ids=prs.slides._sldIdLst;last=ids[-1];ids.remove(last);ids.insert(2,last)
    for index,slide in enumerate(prs.slides,1):
        if index in (4,5):
            for shape in slide.shapes:
                if shape.shape_id==114:replace_text_preserving_style(shape,str(index))
    manifest_path=PROJECT/'analysis/video_embedding.json'
    output=(PROJECT/json.loads(manifest_path.read_text())['output'] if manifest_path.exists()
        else PROJECT/'exports'/('田家鑫_10.1-10.7_组会汇报_5页_含完整PPO视频_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.pptx'))
    prs.save(output)
    with zipfile.ZipFile(output) as z:
        embedded=[n for n in z.namelist() if n.startswith('ppt/media/') and n.endswith('.mp4')]
        assert len(embedded)==1
        assert hashlib.sha256(z.read(embedded[0])).hexdigest()==hashlib.sha256(VIDEO.read_bytes()).hexdigest()
    assert len(Presentation(output).slides)==5
    manifest=dict(source=str(SOURCE.relative_to(PROJECT)),output=str(output.relative_to(PROJECT)),
        video=str(VIDEO.relative_to(PROJECT)),poster=str(POSTER.relative_to(PROJECT)),
        embedded_media=embedded[0],video_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
        slide_count=5,video_slide=3,original_slides_preserved_except_page_numbers=True,
        playback='Click video poster in PowerPoint slideshow; embedded H264 MP4, no external dependency.')
    (PROJECT/'analysis/video_embedding.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    main()
