"""固定官方仓库版本，下载只读研究快照；不导入外部代码或启动仿真。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'docs/lite3-multiterrain-research-20261116'
SOURCES={
    'legged_gym':('leggedrobotics/legged_gym','8fa29acc6fd1910c3d9659eef6310bdd301cde0a',[
        'README.md','legged_gym/envs/base/legged_robot_config.py','legged_gym/envs/base/legged_robot.py',
        'legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py','legged_gym/utils/terrain.py']),
    'himloco':('InternRobotics/HIMLoco','ef289acaa62795009363b7b819c9186690630441',[
        'README.md','projects/himloco/README.md','legged_gym/legged_gym/envs/base/legged_robot_config.py',
        'legged_gym/legged_gym/envs/go1/go1_config.py','legged_gym/legged_gym/envs/base/legged_robot.py',
        'legged_gym/legged_gym/utils/terrain.py','rsl_rl/rsl_rl/modules/him_actor_critic.py',
        'rsl_rl/rsl_rl/modules/him_estimator.py']),
    'walk_these_ways':('Improbable-AI/walk-these-ways','0e7236bdc81ce855cbe3d70345a7899452bdeb1c',[
        'README.md','scripts/train.py','go1_gym/envs/base/legged_robot_config.py','go1_gym/envs/go1/go1_config.py',
        'go1_gym/envs/rewards/corl_rewards.py','go1_gym/utils/terrain.py']),
    'isaaclab_v2_3_0':('isaac-sim/IsaacLab','3c6e67bb5c7ada942a6d1884ab69338f57596f77',[
        'source/isaaclab/isaaclab/terrains/config/rough.py','source/isaaclab/isaaclab/terrains/terrain_generator.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/agents/rsl_rl_ppo_cfg.py']),
}


def main():
    snapshot=Path(tempfile.mkdtemp(prefix='lite3-multiterrain-research-'))
    jobs=[(name,repo,commit,path) for name,(repo,commit,paths) in SOURCES.items() for path in paths]
    def fetch(job):
        name,repo,commit,path=job
        url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
        with urlopen(Request(url,headers={'User-Agent':'Lite3-source-audit'}),timeout=45) as response:
            content=response.read()
        text=content.decode('utf-8');dest=snapshot/name/path
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        return dict(project=name,repository=repo,commit=commit,path=path,
            url=f'https://github.com/{repo}/blob/{commit}/{path}',raw_url=url,
            sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),lines=len(text.splitlines()),
            snapshot_path=str(dest))
    with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(fetch,jobs))
    DOCS.mkdir(parents=True,exist_ok=True)
    (DOCS/'upstream_manifest.json').write_text(json.dumps(dict(date='2026-11-16',
        selection='固定审核版本；Isaac Lab 使用 v2.3.0，其余固定到清单中的提交，不宣称最新版本或各论文实验的原始提交。',
        downloaded_source_executed=False,files=records),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(snapshot=str(snapshot),files=len(records),manifest=str(DOCS/'upstream_manifest.json')),ensure_ascii=False))


if __name__=='__main__':main()
