"""PHUMA data.zip 목록 집계 (압축을 풀지 않고 zipfile 중앙 디렉터리만 읽는다).

실행: .venv/bin/python -I analysis/extra_zip_inventory.py <data.zip 경로> <결과 json 경로>
출력: 폴더별 파일 수, 확장자, 크기 합계, 로봇 폴더 목록. 모션 값은 읽지 않는다.
"""
import collections
import json
import sys
import zipfile


def main(zip_path, out_path):
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
    by_top = collections.Counter()
    by_ext = collections.Counter()
    size_by_ext = collections.Counter()
    dirs = collections.Counter()
    for info in infos:
        if info.is_dir():
            continue
        parts = info.filename.split("/")
        top = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1])
        by_top[top] += 1
        ext = info.filename.rsplit(".", 1)[-1].lower() if "." in parts[-1] else "(none)"
        by_ext[ext] += 1
        size_by_ext[ext] += info.file_size
        dirs[parts[0]] += 1
    summary = {
        "n_members": len(infos),
        "top_level_counts": dict(dirs),
        "files_by_ext": dict(by_ext),
        "uncompressed_bytes_by_ext": dict(size_by_ext),
        "files_by_folder_prefix_first3": dict(sorted(by_top.items())),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
