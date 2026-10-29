# Lightning Grasp (arXiv:2511.07418) 분석 노트

*Lightning Grasp: High Performance Procedural Grasp Synthesis with Contact Fields*
(Zhao-Heng Yin, Pieter Abbeel, UC Berkeley, 2025)를 논문 표 재계산, 공개 코드
[zhaohengyin/lightning-grasp](https://github.com/zhaohengyin/lightning-grasp)(`af43818`) 대조, 안정성 점수식과
contact field 표본 밀도의 CPU toy로 분석한 자료입니다. 공식 코드는 NVIDIA GPU 전용이라 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 표 |
| `analysis/03_analysis.md` | FSWO 점수의 크기 의존성·PGD 정확도·warm start·좌표계 혼용, contact field 밀도, 처리량 산술 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D13), 수치 재계산(C1–C7, V1–V3), 구현 가능성 |
| `analysis/table_check.py` | Table 1·Fig. 1·Fig. 10·본문 수치 재계산 |
| `analysis/fswo_toy.py` | 공식 코드의 안정성 점수식을 numpy로 옮겨 scipy NNLS와 비교 |
| `analysis/contact_field_toy.py` | 공식 코드 설정의 contact field box당 벡터 수 (Allegro 비슷한 손가락 가정) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/fswo_toy.py && python3 analysis/contact_field_toy.py
cd analysis && python3 plot_figures.py
```

모두 CPU 단일 스레드로 30초 안에 끝납니다. 논문 PDF나 공식 코드는 필요하지 않습니다.

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 공식 코드는 CC BY-NC 4.0입니다. 이 저장소의 스크립트는 그 식(안정성 점수, 기본 설정값)을 다시 구현한 것이고 코드를 복사하지 않았습니다.
- `throughput.png`는 논문 Table 1·Fig. 1 수치로, `fswo_scale.png`와 `cf_occupancy.png`는 직접 만든 toy로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
