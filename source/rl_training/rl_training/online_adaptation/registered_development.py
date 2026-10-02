"""Explicit new-development membership for locked frozen route-transfer evaluation."""
from .terrain_protocol import TerrainGeometry, default_flat_online_protocol

REGISTERED_DEVELOPMENT_SEEDS = (834818272, 1490653948, 1132853762, 874689437)
REGISTRATION_SHA256 = "d09d98849d0808c822aecf4d774ceba5d7a6e03b1a5745ff16e3f54c9c52eb77"


def registered_development_geometry(stage_name, seed):
    if type(seed) is not int or seed not in REGISTERED_DEVELOPMENT_SEEDS:
        raise ValueError('Seed is not in registered-development membership')
    if stage_name != 'rough_4cm':
        raise ValueError('Registered development only permits the locked rough_4cm geometry')
    legacy = default_flat_online_protocol()
    if set(REGISTERED_DEVELOPMENT_SEEDS) & (set(legacy.development_seeds) | set(legacy.sealed_seeds)):
        raise ValueError('Registered terrain membership overlaps legacy or sealed seeds')
    return TerrainGeometry(step_height=0.0, rough_amplitude=0.04, seed=seed)
