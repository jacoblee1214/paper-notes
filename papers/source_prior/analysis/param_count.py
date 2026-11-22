"""LeaP 구성요소별 파라미터 수를 코드의 레이어 모양(shape)에서 직접 계산한다.
공식 코드를 import하거나 실행하지 않는다. 레이어 크기는 configs/leap.yaml과
policy.py·flow_net.py·pointnet.py를 읽고 옮겨 적은 것이다.
실행: python3 param_count.py
"""
import json

def lin(i, o): return i * o + o
def ln(d, affine=True): return 2 * d if affine else 0

da, H, To, E = 14, 8, 3, 128   # RoboTwin aloha-agilex 14축, horizon 8, 관측 3, encoder dim 128
ds = 14                        # agent_pos(관절 위치) 차원
hid, L, ratio, tdim = 512, 4, 4.0, 256

pc = (lin(3, 64) + ln(64) + lin(64, 128) + ln(128) + lin(128, 256) + ln(256)
      + lin(256, E) + ln(E))
state_enc = lin(ds, 128) + lin(128, E)
head = ln(E * To) + lin(E * To, 512) + lin(512, 2 * da)
proj = 2 * (lin(H * da, 128) + lin(128, E))
cond_dim = 2 * E * To
flow = (lin(H * da, hid)
        + lin(tdim, tdim * 4) + lin(tdim * 4, hid)
        + lin(cond_dim, hid)
        + L * (lin(hid, int(hid * ratio)) + lin(int(hid * ratio), hid) + lin(hid, 3 * hid))
        + ln(hid) + lin(hid, H * da))

parts = dict(pc_encoder=pc, state_encoder=state_enc, prior_head=head,
             align_proj=proj, flow_net=flow)
total = sum(parts.values())
noprior_infer = pc + state_enc + flow          # NoPrior: prior head, 정렬 투영 없음
leap_infer = noprior_infer + head              # 정렬 투영 제외(추론에 불필요)
res = dict(parts=parts, total_with_align=total, leap_without_align_proj=leap_infer,
           noprior=noprior_infer, paper_leap_M=13.22, paper_noprior_M=13.01,
           head_over_leap_without_proj_pct=100 * head / leap_infer)
for k, v in parts.items(): print(f"{k:14s} {v:>10,d}  ({v/1e6:.4f}M)")
print(f"LeaP total(투영 포함) {total/1e6:.4f}M | 투영 제외 {leap_infer/1e6:.4f}M | NoPrior {noprior_infer/1e6:.4f}M")
print(f"prior head 비율 {res['head_over_leap_without_proj_pct']:.2f}%")
json.dump(res, open("param_count_results.json", "w"), indent=1)
