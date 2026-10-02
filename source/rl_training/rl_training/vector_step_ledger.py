"""Conservative per-vector-step charging for separately budgeted policy training."""
from pathlib import Path
from rl_training.online_adaptation.provenance import write_json_atomic


class VectorStepLedger:
    def __init__(self,path,*,num_envs,max_transitions,metadata):
        self.path=Path(path)
        if self.path.exists():raise ValueError('ledger exists; never silently restart a training attempt')
        if type(num_envs) is not int or num_envs<1 or type(max_transitions) is not int or max_transitions<num_envs:
            raise ValueError('positive vector width and sufficient integer budget required')
        self.state={'schema':'lite3_vector_step_ledger_v1','status':'running','num_envs':num_envs,
                    'max_transitions':max_transitions,'charged':0,'confirmed':0,'pending':None,'metadata':metadata}
        self._save()

    def _save(self):write_json_atomic(self.path,self.state)

    def reserve(self):
        s=self.state
        if s['status']!='running' or s['pending'] is not None:raise ValueError('step already pending or ledger closed')
        if s['charged']+s['num_envs']>s['max_transitions']:raise ValueError('vector step exceeds registered budget')
        s['charged']+=s['num_envs'];s['pending']=s['charged']//s['num_envs'];self._save()

    def confirm(self):
        s=self.state
        if s['status']!='running' or s['pending'] is None:raise ValueError('no pending vector step')
        s['confirmed']+=s['num_envs'];s['pending']=None;self._save()

    def finish(self,*,aborted=False):
        if self.state['status']!='running':raise ValueError('ledger already closed')
        if not aborted and self.state['pending'] is not None:raise ValueError('unconfirmed physical step')
        self.state['status']='aborted' if aborted else 'complete'
        self.state['uncertain_transitions']=self.state['charged']-self.state['confirmed'];self._save()


class ChargedVecEnv:
    """Proxy the RSL vector environment, charging all rows before each step."""
    def __init__(self,env,ledger,*,before_step=None,after_step=None):
        self.env=env;self.ledger=ledger;self.before_step=before_step;self.after_step=after_step
        self._observations=None
        if env.num_envs!=ledger.state['num_envs']:raise ValueError('vector width differs from ledger')

    def __getattr__(self,name):return getattr(self.env,name)

    def get_observations(self):
        self._observations=self.env.get_observations()
        return self._observations

    def step(self,actions):
        snapshot=self.before_step(self._observations,actions) if self.before_step else None
        self.ledger.reserve()
        result=self.env.step(actions)
        self.ledger.confirm()
        self._observations=result[0]
        if self.after_step:self.after_step(snapshot,result)
        return result
