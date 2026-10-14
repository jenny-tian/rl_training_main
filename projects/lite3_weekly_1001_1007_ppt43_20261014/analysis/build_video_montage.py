"""Concatenate six preselected clips into three labelled comparison chapters."""
from pathlib import Path
import hashlib
import json
import subprocess

from PIL import Image, ImageDraw, ImageFont

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parents[1]
MEDIA=PROJECT/'media'
MEDIA.mkdir(exist_ok=True)
WORK=PROJECT/'analysis/video_edit'
WORK.mkdir(exist_ok=True)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,duration,nb_frames,avg_frame_rate','-of','json',str(path)]))['streams'][0]


def get_clip(run,model):
    folder=ROOT/'artifacts/full_ppo_video_20261014'/run
    r=json.loads((folder/'report.json').read_text());assert r['status']=='complete'
    phase=r['phases']['course-'+model];i=phase['video']['env_id']
    path=ROOT/phase['video']['path'];assert digest(path)==phase['video']['sha256']
    return dict(path=path,episode=phase['episodes'][i],probe=probe(path),
        initial_sha256=phase['initial_sha256'],run=run,model=model)


def canvas(title,sub,left,right):
    bg=Image.new('RGB',(1440,720),'#102332');draw=ImageDraw.Draw(bg)
    f=ImageFont.truetype(FONT,32,index=2);small=ImageFont.truetype(FONT,23,index=2)
    draw.text((30,17),title,font=f,fill='white')
    draw.text((30,64),sub,font=small,fill='#9adbe7')
    for x,label,clip in [(24,left[0],left[1]),(744,right[0],right[1])]:
        e=clip['episode'];labels={'orientation':'姿态终止','contact':'非法接触','lateral':'横向越界'}
        state='完成路段' if e['success'] else '失败：'+','.join(labels[k] for k,v in e.get('failure_terms',{}).items() if v)
        draw.text((x,580),label+' · '+state,font=small,fill='white')
        draw.text((x,617),f"横偏 RMS {e['body_lateral_rms_m']*100:.1f} cm / 峰值 {e['max_body_lateral_m']*100:.1f} cm",font=small,fill='#bcd3df')
    draw.text((30,674),'10/14 新补录；不是原59/60评测录像。短片结束后保持最后一帧，不展示重置。',font=small,fill='#b6c6d0')
    return bg


def main():
    clips={
        'base101':get_clip('base-101-attempt01','base'),
        'base107':get_clip('base-attempt01','base'),
        'ppo283101':get_clip('ppo283-101-attempt01','ppo283'),
        'ppo283107':get_clip('ppo283-attempt01','ppo283'),
        'ppo307101':get_clip('ppo307-101-attempt01','ppo307'),
        'ppo307107':get_clip('ppo307-107-attempt01','ppo307'),
    }
    assert len({c['initial_sha256'] for c in clips.values()})==1
    groups=[
        ('01 · rough101：原平地 PPO / 完整 PPO 微调','同一初态、执行器延迟与前进命令 · 2倍速',
         ('原平地模型',clips['base101']),('微调模型283',clips['ppo283101'])),
        ('02 · rough107：原平地 PPO / 完整 PPO 微调','同一初态、执行器延迟与前进命令 · 2倍速',
         ('原平地模型',clips['base107']),('微调模型283',clips['ppo283107'])),
        ('03 · 独立微调模型307：两种崎岖地形','独立训练模型的补充演示 · 2倍速',
         ('模型307 / rough101',clips['ppo307101']),('模型307 / rough107',clips['ppo307107'])),
    ]
    chunks=[];timeline=[];offset=0
    for i,(title,sub,left,right) in enumerate(groups):
        bg=canvas(title,sub,left,right);background=WORK/f'chapter-{i+1}.png';bg.save(background)
        duration=max(float(left[1]['probe']['duration']),float(right[1]['probe']['duration']))/2+1.5
        outfile=WORK/f'chapter-{i+1}.mp4'
        filt=('[0:v]format=yuv420p[bg];'
              '[1:v]setpts=(PTS-STARTPTS)/2,scale=720:405,fps=25,tpad=stop_mode=clone:stop_duration=40[l];'
              '[2:v]setpts=(PTS-STARTPTS)/2,scale=720:405,fps=25,tpad=stop_mode=clone:stop_duration=40[r];'
              '[bg][l]overlay=0:140:shortest=1[tmp];[tmp][r]overlay=720:140:shortest=1[v]')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1','-loop','1','-i',str(background),
            '-i',str(left[1]['path']),'-i',str(right[1]['path']),'-filter_complex',filt,'-map','[v]',
            '-t',str(duration),'-an','-r','25','-c:v','libx264','-threads','4','-preset','fast','-crf','21',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(outfile)],check=True)
        actual=float(probe(outfile)['duration']);chunks.append(outfile)
        timeline.append(dict(chapter=i+1,title=title,start_s=offset,duration_s=actual,
            left=dict(run=left[1]['run'],model=left[1]['model']),right=dict(run=right[1]['run'],model=right[1]['model'])))
        offset+=actual
    concat=WORK/'concat.txt';concat.write_text('\n'.join("file '"+str(p)+"'" for p in chunks)+'\n')
    output=MEDIA/'完整PPO微调_六段拼接对比_2倍速_20261014.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),
        '-c','copy','-movflags','+faststart',str(output)],check=True)
    poster=MEDIA/'完整PPO微调_演示封面.png'
    # Capture after16s simulator time so the robot is visibly on rough terrain.
    subprocess.run(['ffmpeg','-y','-loglevel','error','-ss','8','-i',str(output),'-frames:v','1',str(poster)],check=True)
    original=MEDIA/'原速分段';original.mkdir(exist_ok=True)
    import shutil
    segments=[]
    for key,clip in clips.items():
        target=original/(key+'_原速.mp4');shutil.copy2(clip['path'],target)
        assert digest(target)==digest(clip['path'])
        segments.append(dict(key=key,file=str(target.relative_to(PROJECT)),sha256=digest(target),
            episode=clip['episode'],duration_s=float(clip['probe']['duration'])))
    manifest=dict(output=str(output.relative_to(PROJECT)),sha256=digest(output),poster=str(poster.relative_to(PROJECT)),
        probe=probe(output),timeline=timeline,original_segments=segments,playback_rate=2,
        footage='New Oct14 frozen-model runs; original acceptance remains59/60. No seed or model replacement.')
    (PROJECT/'analysis/montage_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(video=str(output),duration_s=offset,chapters=3,source_clips=6),ensure_ascii=False))


if __name__=='__main__':
    main()
