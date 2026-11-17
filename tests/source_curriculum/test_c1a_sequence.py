from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/tools'))
import reference_online_c1a_sequence as sequence
from reference_stairs_eval import save,sha


def prepare(monkeypatch,tmp_path):
    output,docs = tmp_path/'runs',tmp_path/'docs'
    monkeypatch.setattr(sequence,'OUTPUT',output)
    monkeypatch.setattr(sequence,'DOCS',docs)
    monkeypatch.setattr(sequence,'source_manifest',lambda:{'test':'locked'})
    schedule = sequence.jobs()
    save(output/'lock.json',{'test':'locked'})
    save(output/'attempts.json',schedule[:2])
    for job in schedule[:2]:
        name = job['name']
        save(output/name/'report.json',{'status':'passed'})
        save(output/name/'trace_index.json',{'chunks':[]})
        save(docs/f'AUDIT_{name}.json',dict(status='passed',report_sha256=sha(output/name/'report.json'),
                                         trace_index_sha256=sha(output/name/'trace_index.json')))
    monkeypatch.setattr(sequence,'completed',lambda name:None)
    return output,docs,schedule


def test_only_pending_registered_jobs_are_returned(monkeypatch,tmp_path):
    _,_,schedule = prepare(monkeypatch,tmp_path)
    assert sequence.pending_jobs()==schedule[2:]
    assert sequence.pending_jobs()[0]['name']=='train'
    assert len(sequence.pending_jobs())==45


def test_reserved_failed_or_unaudited_job_is_never_retried(monkeypatch,tmp_path):
    output,_,schedule = prepare(monkeypatch,tmp_path)
    save(output/'attempts.json',schedule[:3])
    with pytest.raises(FileNotFoundError): sequence.pending_jobs()
    def failure(name):
        if name=='train': raise RuntimeError('failed run')
    monkeypatch.setattr(sequence,'completed',failure)
    with pytest.raises(RuntimeError,match='failed run'): sequence.pending_jobs()


def test_changed_artifacts_and_reordered_attempts_rejected(monkeypatch,tmp_path):
    output,_,schedule = prepare(monkeypatch,tmp_path)
    save(output/'smoke_1/report.json',{'status':'changed'})
    with pytest.raises(RuntimeError,match='stale audit'): sequence.pending_jobs()
    save(output/'attempts.json',schedule[1:2])
    with pytest.raises(RuntimeError,match='prefix'): sequence.pending_jobs()
