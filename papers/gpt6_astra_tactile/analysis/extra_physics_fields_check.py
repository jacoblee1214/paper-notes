"""UniVTAC Insert Hole 데모 한 에피소드의 HDF5 필드 목록으로, 힘·재질 척도가 있는지 확인한다.

입력: 로컬에 받은 에피소드 HDF5 한 개(경로 인자). 원자료는 결과에 담지 않는다.
출력: analysis/extra_physics_fields_check_results.json (파일 크기, sha256, 데이터셋 이름·형상, 힘 관련 키 검색 결과,
      이미 만든 파생 특징값 개수).

실행: python3 -I analysis/extra_physics_fields_check.py <hdf5 경로>
"""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DERIVED = ROOT / "data" / "derived"
RESULTS = HERE / "extra_physics_fields_check_results.json"
FORCE_WORDS = ("force", "torque", "wrench", "friction", "mu", "modulus", "young", "poisson",
               "stiffness", "load", "normal", "contact")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(path: Path) -> None:
    datasets = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
        f.visititems(visit)
    names = sorted(datasets)
    hits = [n for n in names if any(w in n.lower() for w in FORCE_WORDS)]
    derived = sorted(DERIVED.glob("ep_*.npz"))
    keys = set()
    for p in derived:
        with np.load(p, allow_pickle=False) as z:
            keys.update(z.files)
    out = {
        "file_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "n_datasets": len(names),
        "datasets": datasets,
        "force_like_dataset_names": hits,
        "force_like_found": bool(hits),
        "derived_episodes": len(derived),
        "derived_feature_keys": sorted(keys),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "conclusion": ("힘·재질 관련 데이터셋 없음: 절대 힘, Hertz 힘, slip margin은 이 데이터로 계산할 수 없음"
                       if not hits else "힘 관련 데이터셋 있음: 확인 후 사용 가능 여부를 따로 판단"),
    }
    RESULTS.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("n_datasets", "force_like_found", "derived_episodes", "conclusion")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
