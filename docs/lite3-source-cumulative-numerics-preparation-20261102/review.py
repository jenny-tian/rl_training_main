"""Independently review this preparation archive on CPU, without simulator imports."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    checked = 0
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ',1)
        path = (HERE/name).resolve()
        check(path.is_relative_to(HERE), 'archive path escaped directory')
        check(hashlib.sha256(path.read_bytes()).hexdigest()==expected, 'archive hash differs: '+name)
        checked += 1
    def read(name): return json.loads((HERE/'raw'/name).read_text())
    provenance = read('provenance.json')
    for name, reference in provenance['raw_files'].items():
        data = (HERE/'raw'/name).read_bytes()
        check(len(data)==reference['bytes'] and hashlib.sha256(data).hexdigest()==reference['sha256'],
              'original evidence changed: '+name)
    check((HERE/'raw/production_ledger_before.json').read_bytes()==
          (HERE/'raw/production_ledger_after.json').read_bytes(), 'preparation changed production ledger')
    ledger = read('production_ledger_after.json')
    check(len(ledger['entries'])==1, 'unexpected production history')
    check(ledger['entries'][0]['state']=='accepted' and ledger['entries'][0]['reserved_transitions']==384,
          'accepted diagnostic charge lost')
    check(read('input_hashes_before.json')==read('input_hashes_after.json'), 'production inputs changed during checks')
    check(read('dependencies_before.json')==read('dependencies_after.json'), 'core dependencies changed')
    check(len(read('input_hashes_before.json'))==230, 'input inventory incomplete')
    check(len(read('dependencies_before.json'))==44, 'dependency inventory incomplete')
    prepared = read('proposals/preparation.json')
    check(prepared['formal_launch_ready'] is False and prepared['cuda_initialized'] is False,
          'unqualified preparation reported as ready')
    for field in ('real_source_ppo_updates','isaac_sim_launches','gpu_jobs'):
        check(prepared[field]==0, 'preparation executed runtime work')
    plans = read('proposals/cumulative_budgets.json')
    ceilings = {'numerics':262144,'midpoint':1048576,'pilot':2097152}
    for run in ('1','2','3'):
        baseline = 384 if run=='1' else 0
        check(plans[run]['baseline_charged_transitions']==baseline, 'prior charge changed')
        for name,ceiling in ceilings.items():
            node = plans[run]['nodes'][name]
            updates = (ceiling-baseline)//2880
            check(node==dict(ceiling=ceiling,remaining_updates=updates,planned_transitions=updates*2880,
                cumulative_transitions=baseline+updates*2880,unused_transitions=ceiling-baseline-updates*2880),
                'cumulative node arithmetic differs')
        authorization = read(f'proposals/run{run}.authorization.proposal.json')
        check(authorization['status']=='proposed' and authorization['user_approved'] is False,
              'proposal became approved')
        check(authorization['cumulative_budget']==plans[run], 'proposal budget differs')
        check(authorization['max_transitions']==plans[run]['nodes']['numerics']['planned_transitions'],
              'numeric rollout limit differs')
        for field in ('runtime_qualification','source_selection_lock','metrics_qualification'):
            reference = authorization[field]
            path = HERE/'raw/proposals'/Path(reference['path']).name
            check(hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256'], 'proposal receipt changed')
    selection = read('proposals/source_selection.proposal.json')
    check(selection['status']=='proposed' and selection['user_approved'] is False, 'selection lock unexpectedly approved')
    check(len(selection['registered_cells'])==15 and len(selection['seed_splits'])==15, 'Source cells lost')
    check(selection['evaluation_episodes_max']['total']==910, 'evaluation budget differs')
    check(read('proposals/runtime_readiness.not_run.json')['status']=='not_run', 'runtime qualification invented')
    check(read('proposals/metrics_qualification.not_run.json')['live_normalized_metric_collector_qualified'] is False,
          'metric qualification invented')
    execution = read('regression_execution.json')
    check(execution['returncode']==0 and execution['environment_overrides']['CUDA_VISIBLE_DEVICES']=='',
          'registered regression failed or GPU mask differs')
    suite = ET.parse(HERE/'raw/regression.junit.xml').getroot().find('testsuite')
    check(suite is not None and int(suite.attrib['tests'])==542, 'registered test count differs')
    check(all(int(suite.attrib[key])==0 for key in ('errors','failures','skipped')), 'registered regression incomplete')
    supplementary = ET.parse(HERE/'raw/final_cli.junit.xml').getroot().find('testsuite')
    check(supplementary is not None and int(supplementary.attrib['tests'])==3,
          'supplementary CLI checks missing')
    check(all(int(supplementary.attrib[key])==0 for key in ('errors','failures','skipped')),
          'supplementary CLI checks failed')
    check((HERE/'raw/checkpoint_reference.json').is_file(), 'original checkpoint reference missing')
    return dict(status='passed_cpu_preparation_archive_review',hashed_files=checked,
        registered_tests=542,supplementary_cli_tests=3,distinct_tests=545,
        production_inputs=230,core_dependencies=44,
        production_budget_unchanged=True,real_source_transitions=0,real_source_ppo_updates=0,
        isaac_sim_launches=0,gpu_jobs=0,formal_launch_ready=False)


if __name__=='__main__':
    result = main()
    (HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
