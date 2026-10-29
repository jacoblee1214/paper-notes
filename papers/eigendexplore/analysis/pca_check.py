"""공개 PCA 자산(repo/src/action_bench/assets/*/pca.safetensors) 검산과 벤치마크별 초기 탐색 공분산 비교.
실행: python3 analysis/pca_check.py   (repo/ 에 공식 코드 shallow clone이 있어야 함, README 참조)

- P1: 손별 90% 분산 k, 상위 3개 성분 누적 분산 (본문 Sec. 3.1, 4.2와 대조)
- P2: 설정별 초기 공분산 Σ = diag(σ_q²) + Eᵀ diag(σ_e²) E (정규화 관절 좌표)의
      관절당 평균 분산(Joint-Space = 1), k차원 부분공간 안의 분산 비율, 참여비 유효 차원
      - 논문 서술(Sec. 3.2): 총 분산을 Joint-Space와 같게, 둘로 반씩, σ_e ∝ √λ
      - 공개 DeXtreme preset: iid √0.5, eigen RMS √0.5, spectrum_power 1
      - 공개 SimToolReal preset: iid 1, eigen RMS 1, spectrum_power 0.5
- P3: SPIDER 설정의 스펙트럼(σ_e ∝ λ)과 기존 IID 대비 총 분산 (가정 명시)
- P4: 오차 방향 정렬 이득 uᵀΣu / (tr Σ / d): 사람 자세 분포에서 뽑은 오차 방향 vs 등방 방향
"""
import json
import os
import struct
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "repo" / "src" / "action_bench" / "assets"
HANDS = ["sharpa", "wuji", "allegro", "xhand", "schunk", "inspire"]
PAPER_K = {"sharpa": 9, "xhand": 6, "allegro": 10, "inspire": 3}   # Sec. 3.1, 4.1.2, 4.2


def load_safetensors(path: Path) -> dict:
    """safetensors 형식(8바이트 길이 + JSON 헤더 + 원시 배열)을 numpy로 읽는다."""
    b = path.read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    head = json.loads(b[8:8 + n])
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        dt = {"F32": np.float32, "F64": np.float64}[v["dtype"]]
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(b[8 + n + s:8 + n + e], dtype=dt).reshape(v["shape"]).astype(np.float64)
    return out


def hand(h: str) -> dict:
    t = load_safetensors(ASSETS / h / "pca.safetensors")
    m = yaml.safe_load((ASSETS / h / "manifest.yaml").read_text())
    half = (np.array(m["joint_upper"]) - np.array(m["joint_lower"])) / 2
    return {"t": t, "m": m, "half": half, "names": m["joint_names"]}


def k_at(ev: np.ndarray, rho: float) -> int:
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), rho)) + 1


def eigen_cov(H: dict, k: int, eigen_rms: float, power: float) -> tuple[np.ndarray, np.ndarray]:
    """distribution.EigenDExplore.from_pca와 같은 계산: 기저를 반범위로 나누고 σ_e를 RMS에 맞춘다."""
    B = H["t"]["components"][:k] / H["half"]                 # (k, d) 정규화 좌표
    sig = H["t"]["component_std"][:k] ** power
    rms = np.sqrt(((B * sig[:, None]) ** 2).sum(0).mean())
    sig = sig * eigen_rms / rms
    return B.T @ np.diag(sig ** 2) @ B, B


def stats(S: np.ndarray, B: np.ndarray) -> dict:
    d = S.shape[0]
    Q, _ = np.linalg.qr(B.T)                                  # 부분공간 정규직교 기저 (d, k)
    inside = float(np.trace(Q.T @ S @ Q) / np.trace(S))
    w = np.linalg.eigvalsh(S)
    return {
        "mean_var_per_joint": round(float(np.trace(S) / d), 3),
        "share_in_k_subspace": round(inside, 3),
        "share_if_isotropic": round(B.shape[0] / d, 3),
        "participation_dim": round(float(w.sum() ** 2 / (w ** 2).sum()), 2),
        "d": d,
        "per_joint_std_min_max": [round(float(np.sqrt(S.diagonal()).min()), 3), round(float(np.sqrt(S.diagonal()).max()), 3)],
    }


def alignment_gain(S: np.ndarray, C_h: np.ndarray, rng, n=4000) -> dict:
    """오차 방향 u에서 uᵀΣu / (trΣ/d). ES·CEM 한 걸음은 대략 Σ∇f라서 이 값이 그 방향의 상대 보폭이다."""
    d = S.shape[0]
    L = np.linalg.cholesky(C_h + 1e-9 * np.eye(d))
    out = {}
    for name, U in (("human_error_dirs", rng.standard_normal((n, d)) @ L.T), ("isotropic_dirs", rng.standard_normal((n, d)))):
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        g = np.einsum("ni,ij,nj->n", U, S, U) / (np.trace(S) / d)
        out[name] = round(float(g.mean()), 3)
    return out


def main():
    rng = np.random.default_rng(0)
    res = {"P1_k_and_variance": {}, "P2_initial_covariance": {}, "P4_alignment_gain": {}}
    hs = {h: hand(h) for h in HANDS}
    catalog = yaml.safe_load((ASSETS / "catalog.yaml").read_text())["hands"]
    for h, H in hs.items():
        ev = H["t"]["explained_variance"]
        k = k_at(ev, 0.9)
        cum = np.cumsum(ev) / ev.sum()
        res["P1_k_and_variance"][h] = {
            "d": len(ev), "k90": k, "catalog_k": catalog[h]["k"], "paper_k": PAPER_K.get(h),
            "cum_at_k90": round(float(cum[k - 1]), 3), "cum_top3": round(float(cum[2]), 3),
            "human_total_var_rad2": round(float((H["t"]["component_std"] ** 2).sum()), 3),
            "human_joint_rms_rad": round(float(np.sqrt((H["t"]["component_std"] ** 2).sum() / len(ev))), 3),
            "fit_corpora": H["m"]["source_corpora"], "fit_dataset_weights": H["m"]["fit"]["dataset_weights"],
            "fit_source_sequences": H["m"]["fit_summary"]["source_sequence_count"],
        }
    # P2: 설정별 초기 공분산
    configs = {
        "paper_text (matched, split, √λ)": (np.sqrt(0.5), np.sqrt(0.5), 1.0),
        "release DeXtreme preset": (np.sqrt(0.5), np.sqrt(0.5), 1.0),
        "release SimToolReal preset": (1.0, 1.0, 0.5),
        "joint_space baseline": (1.0, None, None),
    }
    for h in ("sharpa", "allegro"):
        H = hs[h]
        k = catalog[h]["k"]
        d = len(H["half"])
        row = {}
        for name, (iid, rms, power) in configs.items():
            Se, B = eigen_cov(H, k, rms if rms else 1.0, power if power else 1.0)
            S = iid ** 2 * np.eye(d) + (Se if rms else 0.0)
            row[name] = stats(S, B)
        res["P2_initial_covariance"][h] = row
        # 정규화 좌표에서 eigen 잡음이 큰 관절 (반범위가 작은 외전 관절이 커지는지)
        Se, _ = eigen_cov(H, k, np.sqrt(0.5), 1.0)
        sd = np.sqrt(Se.diagonal())
        order = np.argsort(-sd)
        row["eigen_std_top3_joints(normalized)"] = [[H["names"][i], round(float(sd[i]), 3), round(float(H["half"][i]), 3)] for i in order[:3]]
        row["eigen_std_bottom3_joints(normalized)"] = [[H["names"][i], round(float(sd[i]), 3), round(float(H["half"][i]), 3)] for i in order[-3:]]
        # P4: 사람 자세 공분산(정규화 좌표, 전 성분)
        Bf = H["t"]["components"] / H["half"]
        C_h = Bf.T @ np.diag(H["t"]["component_std"] ** 2) @ Bf
        res["P4_alignment_gain"][h] = {}
        for name, (iid, rms, power) in configs.items():
            Se, _ = eigen_cov(H, k, rms if rms else 1.0, power if power else 1.0)
            S = iid ** 2 * np.eye(d) + (Se if rms else 0.0)
            res["P4_alignment_gain"][h][name] = alignment_gain(S, C_h, rng)
    # P4b: 여섯 손 모두, 논문 서술 설정에서 사람 방향 정렬 이득과 부분공간 집중도 (SPIDER Δcost와 비교용)
    res["P4_alignment_gain_all_hands"] = {}
    for h, H in hs.items():
        k = catalog[h]["k"]
        d = len(H["half"])
        Se, B = eigen_cov(H, k, np.sqrt(0.5), 1.0)
        S = 0.5 * np.eye(d) + Se
        Bf = H["t"]["components"] / H["half"]
        C_h = Bf.T @ np.diag(H["t"]["component_std"] ** 2) @ Bf
        g = alignment_gain(S, C_h, rng)
        g.update(stats(S, B))
        res["P4_alignment_gain_all_hands"][h] = g
    # P3: SPIDER (radian 좌표). sampling._powered_component_scale: σ_e ∝ λ (분산 비율의 1제곱)
    H = hs["sharpa"]
    spider = yaml.safe_load((ROOT / "repo/src/action_bench/benchmarks/spider/configs/hands.yaml").read_text())["hands"]["sharpa"]
    k = k_at(H["t"]["explained_variance"], 0.9)
    lam = H["t"]["explained_variance"][:k]
    shapes = {"SPIDER σ_e ∝ λ": lam / lam[0], "paper/DeXtreme σ_e ∝ √λ": np.sqrt(lam / lam[0]),
              "SimToolReal σ_e ∝ λ^¼": (lam / lam[0]) ** 0.25}
    native_sigma_last_knot = 0.1 * 4.0      # runner.overrides: joint_noise_scale 0.1 × last_ctrl_noise_scale 4.0 (가정: 손가락 노이즈 = 이 곱)
    eig_rms = spider["eigen_noise_rms"]       # 스케줄 비율이 같다고 보면 마지막 knot에서 이 RMS
    res["P3_spider"] = {
        "sigma_e_last_over_first": {n: round(float(s[-1]), 3) for n, s in shapes.items()},
        "eigen_noise_rms_rad": round(eig_rms, 4),
        "assumed_native_iid_sigma_rad_last_knot": native_sigma_last_knot,
        "total_var_ratio_vs_iid_only": round((native_sigma_last_knot ** 2 + eig_rms ** 2) / native_sigma_last_knot ** 2, 3),
        "note": "iid_scale 1.0이라 기존 IID는 그대로 두고 eigen을 더한다. 손가락 노이즈가 joint_noise_scale × 스케줄이라는 가정 아래 계산",
    }
    path = ROOT / "analysis" / "pca_check_results.json"
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
