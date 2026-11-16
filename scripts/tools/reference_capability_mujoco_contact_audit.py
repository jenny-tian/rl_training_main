"""Inspect saved poses for foot/shank terrain contacts; no time integration."""
from collections import Counter
import numpy as np
import mujoco

from reference_capability_mujoco import (OUTPUT, DOCS, read, save, sha, make_model,
    terrain_kind, verify_sources, geometric_clear, termination, PERIOD)


def support_from_contacts(contact_list, radius):
    """Sum contact load per distal foot region; exclude distant shank contacts."""
    loads = Counter()
    for leg, kind, distance, force in contact_list:
        if kind == 'FOOT' or (kind == 'SHANK' and distance <= radius):
            loads[leg] += force
    return sum(force >= 1. for force in loads.values())


def main():
    plan = read(OUTPUT/'plan.json'); verify_sources(plan)
    records = read(OUTPUT/'complete.json')['records']
    output = []
    for record in records:
        model = make_model(terrain_kind(record['scene'])); data = mujoco.MjData(model)
        path = OUTPUT/'episodes'/f'{record["policy"]}_{record["scene"]}_{record["repeat"]}'/'trace.npz'
        assert sha(path) == record['trace_sha256']
        trace = np.load(path)
        rows = []
        distal_distances = []
        foot_ids = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{leg}_FOOT_collision')
            for leg in ('FL', 'FR', 'HL', 'HR')}
        radii = (.025, .03, .04)
        scores = {radius: None for radius in radii}
        dwell = 0.
        # Every saved 20 ms state: no additional policy or physics episode.
        for i in range(len(trace['time'])):
            data.qpos[:] = trace['qpos'][i]; data.qvel[:] = trace['qvel'][i]
            data.ctrl[:] = trace['torque'][i]; data.qacc_warmstart[:] = 0.
            mujoco.mj_forward(model, data)
            forces = Counter(); force = np.zeros(6); old_count = set(); contact_list = []
            for j in range(data.ncon):
                con = data.contact[j]
                # Only world/terrain contact, not self contact.
                # Fixed staircase bodies also belong to the static world weld.
                if model.body_weldid[model.geom_bodyid[con.geom1]] != 0 and model.body_weldid[model.geom_bodyid[con.geom2]] != 0:
                    continue
                mujoco.mj_contactForce(model, data, j, force)
                for geom in (con.geom1, con.geom2):
                    if model.body_weldid[model.geom_bodyid[geom]] != 0:
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                        forces[name] += abs(float(force[0]))
                        if '_FOOT_' in name or '_SHANK_' in name:
                            leg, kind = name.split('_')[:2]
                            distance = float(np.linalg.norm(con.pos-data.geom_xpos[foot_ids[leg]]))
                            contact_list.append((leg, kind, distance, abs(float(force[0]))))
                            if kind == 'SHANK' and abs(force[0]) >= 1.:
                                distal_distances.append(distance)
                        if '_FOOT_' in name and abs(force[0]) >= 1.:
                            old_count.add(name)
            foot = sum(v for k, v in forces.items() if '_FOOT_' in k)
            shank = sum(v for k, v in forces.items() if '_SHANK_' in k)
            feet = sum(v >= 1. for k, v in forces.items() if '_FOOT_' in k)
            route = record['scene'] not in ('turn_right', 'arc_left')
            clear = geometric_clear(record['scene'], trace['qpos'][i, :3], trace['feet'][i], record['initial_xyz']) if route else False
            dwell = dwell+PERIOD if clear else 0.
            for radius in radii:
                result = termination(trace['rpy'][i], trace['gravity'][i],
                    trace['qpos'][i, 1]-record['initial_xyz'][1], route, clear, dwell,
                    support_from_contacts(contact_list, radius), bool(trace['torso'][i]))
                if scores[radius] is None and result:
                    scores[radius] = dict(outcome=result, duration_s=float(trace['time'][i]),
                        lateral_m=float(trace['qpos'][i, 1]-record['initial_xyz'][1]))
            rows.append(dict(t=float(trace['time'][i]), foot_force_sum=foot, shank_force_sum=shank,
                foot_supports=feet, original_supports=int(trace['support'][i]),
                per_contact_supports=len(old_count), shank_contact=shank >= 1.))
        active = [r for r in rows if r['t'] > 2.]
        post3 = [r for i, r in enumerate(rows) if trace['qpos'][i, 0]-record['initial_xyz'][0] >= 3.]
        output.append(dict(policy=record['policy'], scene=record['scene'], repeat=record['repeat'],
            sampled_states=len(rows), states_after_2s=len(active),
            shank_contact_fraction_after_2s=sum(r['shank_contact'] for r in active)/len(active),
            two_foot_support_fraction_after_2s=sum(r['foot_supports'] >= 2 for r in active)/len(active),
            reconstructed_original_support_mismatches=sum(r['per_contact_supports'] != r['original_supports'] for r in rows),
            aggregation_changes_foot_count=sum(r['foot_supports'] != r['per_contact_supports'] for r in rows),
            mean_foot_force_sum=float(np.mean([r['foot_force_sum'] for r in active])),
            mean_shank_force_sum=float(np.mean([r['shank_force_sum'] for r in active])),
            loaded_shank_contact_distance_quantiles_m=np.quantile(distal_distances, [0., .5, .95, 1.]).tolist() if distal_distances else [],
            distal_contact_sensitivity={str(radius): score if score is not None else
                dict(outcome=record['outcome'], duration_s=record['duration_s'], lateral_m=record['lateral_m'])
                for radius, score in scores.items()},
            post_3m_samples=len(post3), post_3m_max_foot_supports=max((r['foot_supports'] for r in post3), default=None),
            original_outcome=record['outcome']))
        trace.close()
    result = dict(status='completed', physics_steps=0, policy_steps=0, optimizer_updates=0,
        method='mj_forward on saved qpos/qvel/last applied torque, warmstart reset; contact force estimates, not time integration',
        interpretation='Shank contacts refer to the deployment MJCF collision mesh, not proof of physical robot shank scraping.',
        correction='Use body_weldid to include fixed stair bodies as terrain; initial worldbody-only audit preserved separately.',
        sensitivity='FOOT contact or SHANK contact within 2.5/3/4 cm of corresponding foot center; summed normal load per leg >=1N. All radii reported, not selected by outcome. Original formal results and videos remain unmodified.',
        formal_outcomes_unchanged=True, source_sha256=sha(__file__), records=output)
    save(OUTPUT/'contact_audit.json', result); save(DOCS/'contact_audit.json', result)
    print('CONTACT AUDIT COMPLETE: saved states only; formal results unchanged')


if __name__ == '__main__':
    main()
