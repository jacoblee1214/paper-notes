"""실험 1·2의 "가중치 읽기"가 얼마나 어려운 문제인지 본다: LLM 없이 numpy 통계 몇 줄로 무엇이 보이는가.

- 실험 1: 첫 층 W1(500x32)을 칸(10x10) x 채널(5)로 접어 agent 채널의 가중치 에너지 지도, 출력층 W3(32x5)의 행동별 크기
- 실험 2: 출력층 편향 b3(144)와 출력 가중치 W3(64x144)의 각 행을 12x12 그림으로 접기
forward pass는 하지 않는다(블로그 실험과 같은 조건). 실행: python3 analysis/weight_reading_toy.py [logs_official]
→ analysis/weight_reading_toy_results.json, report/figures/exp1_agent_maps.png, report/figures/exp2_output_units.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
L = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'logs_official')
FIG = os.path.join(HERE, '..', 'report', 'figures')
os.makedirs(FIG, exist_ok=True)
FONT = next((f for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                          '/System/Library/Fonts/AppleSDGothicNeo.ttc'] if os.path.exists(f)), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False
out = {}

# ---------------------------------------------------------------- 실험 1
key = json.load(open(os.path.join(L, 'exp1_which_game/eval/answer_key.json')))
order = ['breakout', 'pong', 'pacman', 'freeway']
ids = {v['game']: k for k, v in key.items()}
ACT = ['no-op', 'up', 'down', 'left', 'right']
fig, ax = plt.subplots(2, 4, figsize=(11, 5.6))
e1 = {}
for j, g in enumerate(order):
    z = np.load(os.path.join(L, f'exp1_which_game/networks/{ids[g]}.npz'))
    W1 = z['actor/dense1/kernel'].reshape(10, 10, 5, 32)        # (행, 열, 채널, 은닉)
    W3 = z['actor/logits/kernel']                              # (32, 5)
    agent = (W1[:, :, 0, :] ** 2).sum(-1)                      # agent 채널 칸별 에너지
    agent_in = agent[1:9, 1:9] / agent[1:9, 1:9].sum()
    ax[0, j].imshow(agent, cmap='Greys')
    ax[0, j].set_title(f'{g}: agent 채널 W1 에너지', fontsize=9)
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    # 행동 로짓의 "구분 정도": 각 행동 열에서 평균을 뺀 크기
    W3c = W3 - W3.mean(1, keepdims=True)
    an = np.linalg.norm(W3c, axis=0)
    ax[1, j].bar(ACT, an / an.max(), color='#4a6fa5')
    ax[1, j].set_title('출력층 행동별 크기(정규화)', fontsize=9)
    ax[1, j].tick_params(axis='x', labelsize=7, rotation=45)
    e1[g] = {'agent_row_share': [round(float(x), 3) for x in agent_in.sum(1)],
             'agent_col_share': [round(float(x), 3) for x in agent_in.sum(0)],
             'top_row': int(np.argmax(agent_in.sum(1))) + 1, 'top_row_share': round(float(agent_in.sum(1).max()), 3),
             'top_col': int(np.argmax(agent_in.sum(0))) + 1, 'top_col_share': round(float(agent_in.sum(0).max()), 3),
             'action_norm': dict(zip(ACT, [round(float(x), 3) for x in an / an.max()]))}
fig.suptitle('실험 1: 학습된 정책의 가중치를 numpy로 접어 본 모습 (forward pass 없음)', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'exp1_agent_maps.png'), dpi=160)
out['exp1'] = e1

# ---------------------------------------------------------------- 실험 2
k2 = json.load(open(os.path.join(L, 'exp2_diffusion/eval/answer_key.json')))
ids2 = {(v['shape'] if isinstance(v, dict) else v): k for k, v in k2.items() if k != 'quality'}
fig, ax = plt.subplots(3, 7, figsize=(11, 5.4))
e2 = {}
for i, s in enumerate(['circle', 'square', 'triangle']):
    z = np.load(os.path.join(L, f'exp2_diffusion/networks/{ids2[s]}.npz'))
    b3, W3 = z['b3'].reshape(12, 12), z['W3']
    ax[i, 0].imshow(b3, cmap='Greys_r'); ax[i, 0].set_ylabel(s, fontsize=9)
    nrm = np.linalg.norm(W3, axis=1)
    top = np.argsort(-nrm)[:6]
    for c, u in enumerate(top):
        ax[i, c + 1].imshow(W3[u].reshape(12, 12), cmap='RdBu_r', vmin=-np.abs(W3[u]).max(), vmax=np.abs(W3[u]).max())
    for a in ax[i]:
        a.set_xticks([]); a.set_yticks([])
    # 편향 그림의 위·아래 폭 비교: 삼각형은 위가 좁다
    m = b3[1:11, 1:11] > b3[1:11, 1:11].mean()
    rows = m.sum(1)
    e2[s] = {'bias_width_top3': round(float(rows[:3].mean()), 2), 'bias_width_bottom3': round(float(rows[-3:].mean()), 2),
             'lr_symmetry_corr': round(float(np.corrcoef(b3.ravel(), b3[:, ::-1].ravel())[0, 1]), 3),
             'top_unit_norms': [round(float(nrm[u]), 2) for u in top]}
ax[0, 0].set_title('출력 편향 b3', fontsize=9)
for c in range(6):
    ax[0, c + 1].set_title(f'W3 행 #{c + 1}', fontsize=9)
fig.suptitle('실험 2: DDPM 출력층을 12x12로 접은 그림 (크기 순 상위 6개 은닉 단위)', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'exp2_output_units.png'), dpi=160)
out['exp2'] = e2

json.dump(out, open(os.path.join(HERE, 'weight_reading_toy_results.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
