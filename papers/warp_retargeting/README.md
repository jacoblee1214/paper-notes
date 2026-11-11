# WARP: Whole-Body Retargeting for Learning from Offline Human Demonstrations (arXiv:2606.29940) 분석 노트

사람 상체 시연을 로봇 전신 행동으로 바꾸는 오프라인 리타게터(WARP)를 표 재계산과 장난감 실험으로 분석한 자료입니다.
논문은 CoRL 2026 표기(프로젝트 페이지 기준)이고, 공개 저장소(MIT)에는 WARP 솔버가 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림·표 제외) |
| `analysis/01_paper_analysis.md` | 방법, 실험, 한계, 질문, v1 → v2 차이 |
| `analysis/02_prior_work.md` | 선행연구, 가까운 대안, GitHub 확인 |
| `analysis/03_analysis.md` | 표 재계산과 검정, 주장 검사, 유도, 장난감 결과 |
| `analysis/04_code_verification.md` | 공개 코드 대조, 주장별 판정, 구현 가능성, 권리 |
| `analysis/table_check.py` | 표 재계산과 검정 (수 초) |
| `analysis/retarget_toy.py` | 장난감 A(평면 다리, 접촉 가중치와 정규화), B(평면 3관절 팔, 자세 가중치와 seed 분산). 약 8초 |
| `analysis/extra_block_causal.py` | 블록 인과 마스크 유무를 같은 데이터에서 비교하는 장난감 (torch 필요, CPU 약 6분) |
| `analysis/plot_figures.py` | 직접 만든 그래프 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/retarget_toy.py
python3 analysis/extra_block_causal.py  # torch 필요, 약 6분
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스는 CC BY-NC-ND 4.0입니다(arXiv abs 페이지). 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 장난감 실험으로 그렸습니다.
- 공개 코드 저장소(MIT)는 `https://github.com/Euler-Rodrigues-Lab/WARP_retargeting`입니다. 이 저장소는 코드를 복사하지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
