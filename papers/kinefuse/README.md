# KineFuse (arXiv:2607.14842) 분석 노트

*KineFuse: Kinematic-Aware Haptic Fusion for In-Hand Occluded-Object Pose Tracking*
(Ahn, Lee, Park, Hwang, **IROS 2026 채택**)를 표 재계산, 노름·드리프트 toy, 구현 가능성 판정으로 분석한 자료입니다.
공식 코드는 없습니다. 저장소 `cold-young/kine-fuse`는 프로젝트 페이지와 README뿐입니다(2026-11-09 확인).

**구현할 수 있는지:** 공식 재현은 불가능합니다(코드·가중치·데이터 미공개, IsaacLab·FoundationPose용 CUDA GPU, 실물 F/T 손 필요).
아이디어 재구현은 구조 자체는 쉽고, 고유감각·근위 F/T·접촉·자세 GT가 함께 있는 데이터가 장벽입니다(`analysis/04_claim_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations, 본문 불일치 목록 |
| `analysis/02_prior_work.md` | 선행연구 계보, 관련 저장소와 GitHub 표 |
| `analysis/03_analysis.md` | Table I·II·III 재계산, 유의성, 유도, toy 결과 |
| `analysis/04_claim_verification.md` | 주장별 판정(C1–C25), 라이선스, 빠진 정보, 구현 가능성 |
| `analysis/table_check.py` | 표 값 재계산, Welch·Holm, 상관, 변동계수 |
| `analysis/norm_dominance_toy.py` | 노름과 attention, ridge 벌점과 스케일, GD 과도 응답 |
| `analysis/drift_toy.py` | 프레임 격차의 chain 증폭, 해석해 |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
python3 analysis/table_check.py && python3 analysis/norm_dominance_toy.py && python3 analysis/drift_toy.py
python3 analysis/plot_figures.py
```

모두 numpy와 scipy로 CPU에서 몇 초 안에 끝납니다. 논문 수치는 스크립트에 표 번호와 함께 적었고, toy의 파라미터는 가정입니다.

## 출처와 라이선스

- 논문은 **CC BY-NC-ND 4.0**입니다(arXiv abs 페이지와 PDF 메타데이터 확인, 본문에 다른 권리 표기 없음). 변경과 상업적 이용이 제한되므로 논문 그림은 이 노트에 싣지 않았습니다.
- `report/figures/`의 그래프는 논문 표 값과 toy 시뮬레이션으로 직접 그린 것입니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
