"""Reuse the validated native-reset environment without changing locked files."""
from contextlib import contextmanager

import reference_online_c1b_env_repaired as environment_module
import reference_online_c1b_env_reset_api as reset_module
from reference_online_c1a import protocol


@contextmanager
def current_protocol():
    # Only used in the single simulator thread. Restore even on exceptions.
    original = environment_module.protocol, reset_module.protocol
    environment_module.protocol = reset_module.protocol = protocol
    try:
        yield
    finally:
        environment_module.protocol, reset_module.protocol = original


class Environment(reset_module.Environment):
    def __init__(self,args,specs):
        with current_protocol():
            super().__init__(args,specs)

    def reset(self,ids,episodes):
        with current_protocol():
            return super().reset(ids,episodes)

    def frame(self):
        result = super().frame()
        result['body_velocity'] = self.robot.data.root_lin_vel_b[:,:2].cpu().numpy().copy()
        return result
