"""공식 코드의 FLUX.2 어텐션 마스크 논리(저장소 imagewam.py의 _build_mot_attention_mask_flux2)를
같은 규칙으로 다시 쓰고, 행동 토큰이 볼 수 있는 토큰 묶음을 확인한다.

저장소를 import하지 않는다(규칙만 옮긴 numpy 사본). 실행: python3 mask_check.py (1초 미만)
확인할 것: 행동 토큰은 텍스트와 참조 이미지(현재 관측) 토큰만 보고, 목표(노이즈) 이미지 토큰은 보지 못한다.
따라서 편집 노이즈 시점 τ가 달라져도 행동 토큰이 읽는 K,V 값은 목표 토큰의 내용에 의존하지 않는다.
"""
import json
import os

import numpy as np


def build_mask(txt, cond, target, action):
    t0, r0, x0, a0 = 0, txt, txt + cond, txt + cond + target
    total = a0 + action
    m = np.zeros((total, total), dtype=bool)
    m[t0:r0, t0:x0] = True        # 텍스트: 텍스트+참조
    m[r0:x0, t0:x0] = True        # 참조: 텍스트+참조
    m[x0:a0, t0:a0] = True        # 목표: 텍스트+참조+목표
    m[a0:total, t0:x0] = True     # 행동: 텍스트+참조
    m[a0:total, a0:total] = True  # 행동: 행동 자기 자신
    return m, (t0, r0, x0, a0, total)


def main():
    txt, cond, target, action = 6, 5, 4, 3
    m, (t0, r0, x0, a0, total) = build_mask(txt, cond, target, action)
    act = m[a0:total]
    out = {
        "행동 토큰이 보는 텍스트 비율": float(act[:, t0:r0].mean()),
        "행동 토큰이 보는 참조 이미지 비율": float(act[:, r0:x0].mean()),
        "행동 토큰이 보는 목표(노이즈) 이미지 비율": float(act[:, x0:a0].mean()),
        "행동 토큰이 보는 행동 비율": float(act[:, a0:total].mean()),
        "텍스트/참조 토큰이 목표를 보는가": bool(m[t0:x0, x0:a0].any()),
        "텍스트/참조 토큰이 행동을 보는가": bool(m[t0:x0, a0:total].any()),
        "목표 토큰이 행동을 보는가": bool(m[x0:a0, a0:total].any()),
    }
    # 추론 때: 목표 토큰 수 0 (infer_action_flux2의 empty_target)
    m0, (_, _, x00, a00, tot0) = build_mask(txt, cond, 0, action)
    out["추론(목표 0개)에서 행동이 보는 접두부 비율"] = float(m0[a00:tot0, :x00].mean())
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_check_results.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
