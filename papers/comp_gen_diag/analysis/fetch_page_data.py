"""프로젝트 페이지 저장소의 공개 결과 파일(CSV, JSON, 문서)을 data/page/에 받는다.

받는 것은 저자가 프로젝트 페이지 저장소에 올린 집계 결과 파일뿐이다(정책 가중치, 시연 데이터, 동영상은 받지 않는다).
  저장소: https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot_Page (main)
  합계 약 3 MB. 이 파일들은 저자의 것이므로 이 저장소(분석 노트)에는 다시 배포하지 않는다(data/는 gitignore).

실행: python3 analysis/fetch_page_data.py   (표준 라이브러리만, 네트워크 필요)
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "page"
BASE = "https://raw.githubusercontent.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot_Page/main/"
FILES = {
    "q1/data/success_vs_b.csv": "q1_data_success_vs_b.csv",
    "q1/data/success_vs_b_ortho.csv": "q1_data_success_vs_b_ortho.csv",
    "q1/data/success_vs_b_task_sets.csv": "q1_data_success_vs_b_task_sets.csv",
    "q1/data/success_vs_b_runs.csv": "q1_data_success_vs_b_runs.csv",
    "q1/data/l4_1s_success_vs_b.csv": "q1_data_l4_1s_success_vs_b.csv",
    "q1/data/l4_1s_success_vs_b_runs.csv": "q1_data_l4_1s_success_vs_b_runs.csv",
    "q1/data/l4_1s_success_vs_b_task_sets.csv": "q1_data_l4_1s_success_vs_b_task_sets.csv",
    "q2/data/ft_success_vs_demos.csv": "q2_data_ft_success_vs_demos.csv",
    "q2/data/random_vs_orthogonal_summary.csv": "q2_data_random_vs_orthogonal_summary.csv",
    "q2/data/success_vs_demos_by_pretrain.csv": "q2_data_success_vs_demos_by_pretrain.csv",
    "q3/36tasks/data/l4_2s_no_cont0_runs.csv": "q3_36tasks_data_l4_2s_no_cont0_runs.csv",
    "q3/36tasks/data/l4_2s_no_cont0_failure_analysis.csv": "q3_36tasks_data_l4_2s_no_cont0_failure_analysis.csv",
    "q3/48tasks/data/l4_2s_no_cont0_runs.csv": "q3_48tasks_data_l4_2s_no_cont0_runs.csv",
    "q3/48tasks/data/l4_2s_no_cont0_failure_analysis.csv": "q3_48tasks_data_l4_2s_no_cont0_failure_analysis.csv",
    "assets/data/modularity_comparison.csv": "assets_data_modularity_comparison.csv",
    "results/data.json": "results_data.json",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for src, name in FILES.items():
        dst = OUT / name
        if dst.exists():
            print("exists", name)
            continue
        with urllib.request.urlopen(BASE + src, timeout=60) as r:
            dst.write_bytes(r.read())
        print("wrote", name, dst.stat().st_size)


if __name__ == "__main__":
    main()
