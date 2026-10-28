# DDPG (arXiv:1509.02971) 분석 노트

*Continuous control with deep reinforcement learning* (Lillicrap, Hunt et al., Google DeepMind, ICLR 2016)을
Table 1 재계산, 주요 공개 구현 4종(Stable-Baselines3, Spinning Up, sfujim/TD3, OpenAI baselines)과의 설정 대조,
numpy로 만든 DDPG toy(진자 swing-up)로 분석한 자료입니다. 공식 코드는 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구·후속 계보(DPG → DDPG → TD3·SAC → FlashSAC), 공개 구현 표 |
| `analysis/03_analysis.md` | Eq. 6의 근사, 표적망 반감기, OU 잡음 dt, 시드 퍼짐 하한, toy 결과 |
| `analysis/04_code_verification.md` | 논문 ↔ 구현 대조(D1–D9), 수치 재계산(C1–C10), 구현 가능성 |
| `analysis/table_check.py` | Table 1·부록 수치 재계산 |
| `analysis/ddpg_toy.py` | numpy DDPG (8개 설정 × 5 시드) |
| `analysis/plot_figures.py` | 그래프와 toy 통계 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/ddpg_toy.py          # 단독 실행이면 수 분, 결과는 ddpg_toy_results.json
cd analysis && python3 plot_figures.py
```

모두 CPU에서 돌아갑니다. 논문 PDF나 다른 저장소는 필요하지 않습니다.

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 Table 1의 수치와 직접 만든 toy 결과로 그렸습니다.
- 공개 구현의 파일 경로와 커밋은 `analysis/02_prior_work.md` §4와 `analysis/04_code_verification.md`에 있습니다. 구현 코드는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
