"""oracle 정책의 privileged info 벡터 배치를 코드(xhand_hora.py)와 같은 규칙으로 재현해 겹침을 점검한다 (CPU, 정적 분석).

규칙 (dexscrew/tasks/xhand_hora.py):
  - L1024-1030: obj_position(0:3), obj_scale(3:4), obj_mass(4:5), obj_friction(5:6), obj_com(6:9) 고정
  - L1032-1057: 나머지 항목을 start_index = 0 부터 다시 이어 붙임 (enable 된 것만)
  - L1059-1065: _update_priv_buf 는 해당 slice 에 그대로 덮어씀
쓰기 시점:
  - 환경 생성 시 1회(L285-358): screw_joint_friction, obj_scale, obj_com, obj_friction, obj_restitution, obj_mass
  - reset 시(L421-546): pgain, dgain, hand_joint_pos, hand_orientation, hand_position, obj_orientation, 그리고 L562 [0:3]=0
  - 매 스텝(L619-628, 이 순서): obj_position, obj_orientation, obj_linvel, fingertip_orientation, fingertip_linvel,
    fingertip_angvel, nut_pos, nut_dof_pos, nut_dof_vel, fingertip_position
  - obj_angvel, hand_scale, nut_contact 는 _update_priv_buf 호출이 없음 → 다른 항목이 덮어쓰지 않으면 0
실행: python3 analysis/priv_layout_check.py [screwdriver|nutbolt]
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = {"screwdriver": "XHandHoraScrewDriver", "nutbolt": "XHandHoraNutBolt"}
FINGERS, NACT = 5, 12


def layout(p):
    flags = {
        "obj_orientation": p["enable_obj_orientation"], "obj_linvel": p["enable_obj_linvel"],
        "obj_angvel": p["enable_obj_angvel"], "fingertip_position": p["enable_ft_pos"],
        "fingertip_orientation": p["enable_ft_orientation"], "fingertip_linvel": p["enable_ft_linvel"],
        "fingertip_angvel": p["enable_ft_angvel"], "hand_scale": p["enable_hand_scale"],
        "obj_restitution": p["enable_obj_restitution"], "tactile": p["enable_tactile"],
        "nut_contact": p["enable_nut_contact"], "nut_pos": p["enable_nut_pos"],
        "nut_dof_vel": p["enable_nut_dof_vel"], "nut_dof_pos": p["enable_nut_dof_pos"],
        "pgain": p["enable_pgain"], "dgain": p["enable_dgain"], "hand_joint_pos": p["enable_hand_joint_pos"],
        "hand_orientation": p["enable_hand_orientation"], "hand_position": p["enable_hand_position"],
        "screw_joint_friction": p.get("enable_screw_joint_friction", False),
    }
    d = OrderedDict([("obj_position", (0, 3)), ("obj_scale", (3, 4)), ("obj_mass", (4, 5)),
                     ("obj_friction", (5, 6)), ("obj_com", (6, 9))])
    dims = OrderedDict([("obj_orientation", 4), ("obj_linvel", 3), ("obj_angvel", 3),
                        ("fingertip_position", 3 * FINGERS), ("fingertip_orientation", 4 * FINGERS),
                        ("fingertip_linvel", 3 * FINGERS), ("fingertip_angvel", 3 * FINGERS),
                        ("hand_scale", 1), ("obj_restitution", 1), ("tactile", 0), ("nut_contact", 1),
                        ("nut_pos", 3), ("nut_dof_vel", 1), ("nut_dof_pos", 1), ("pgain", NACT),
                        ("dgain", NACT), ("hand_joint_pos", NACT), ("hand_orientation", 4),
                        ("hand_position", 3), ("screw_joint_friction", 1)])
    s = 0
    for k, n in dims.items():
        if flags[k]:
            d[k] = (s, s + n)
            s += n
    return d


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "screwdriver"
    cfg = yaml.safe_load((ROOT / "dexscrew" / "configs" / "task" / f"{TASK[task]}.yaml").read_text())
    d = layout(cfg["env"]["privInfo"])
    dim = max(e for _, e in d.values())
    owner = [None] * dim
    init_writes = ["screw_joint_friction", "obj_scale", "obj_com", "obj_friction", "obj_restitution", "obj_mass"]
    reset_writes = ["pgain", "dgain", "hand_joint_pos", "hand_orientation", "hand_position", "obj_orientation"]
    step_writes = ["obj_position", "obj_orientation", "obj_linvel", "fingertip_orientation", "fingertip_linvel",
                   "fingertip_angvel", "nut_pos", "nut_dof_pos", "nut_dof_vel", "fingertip_position"]
    for name in init_writes + reset_writes:
        if name in d:
            s, e = d[name]
            for i in range(s, e):
                owner[i] = name
    for i in range(0, 3):
        owner[i] = "(reset zero)"
    for name in step_writes:
        if name in d:
            s, e = d[name]
            for i in range(s, e):
                owner[i] = name
    report = {"task": task, "priv_info_dim": dim, "layout": {k: list(v) for k, v in d.items()}, "visible": {}}
    for k, (s, e) in d.items():
        kept = sum(1 for i in range(s, e) if owner[i] == k)
        report["visible"][k] = f"{kept}/{e - s}"
    lost = [k for k, v in report["visible"].items() if v.split("/")[0] == "0" and v != "0/0"]
    report["fully_lost"] = lost
    report["owner_by_index"] = owner
    out = ROOT / "analysis" / f"priv_layout_{task}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("task", "priv_info_dim", "visible", "fully_lost")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
