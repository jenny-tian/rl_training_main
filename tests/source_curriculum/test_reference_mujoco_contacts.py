from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_capability_mujoco_contact_audit import support_from_contacts


def test_support_counts_distal_mesh_and_sums_each_leg_once():
    contacts = [('FL', 'FOOT', .022, .6), ('FL', 'SHANK', .024, .6),
        ('HR', 'SHANK', .024, 10.), ('HR', 'SHANK', .024, 8.)]
    assert support_from_contacts(contacts, .025) == 2


def test_proximal_shank_contact_is_not_foot_support():
    contacts = [('FL', 'SHANK', .12, 50.), ('FR', 'FOOT', .022, .4), ('HR', 'SHANK', .024, 10.)]
    assert support_from_contacts(contacts, .04) == 1
