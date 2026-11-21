# 02. 선행연구, 공식 코드, 이 저장소의 관련 분석

## 1. 공식 코드와 프로젝트 페이지

| 항목 | 링크 | 라이선스 | 상태 (2026-11-22 확인) | 처리 |
|---|---|---|---|---|
| 이 논문의 공식 코드 | 없음 | — | 논문 1쪽 각주 "Code to be released." 웹 검색(제목, 저자명, Φ-CON)으로도 찾지 못함 | 복제하지 않음. 정적 대조 대신 `04_claim_verification.md` 작성 |
| 이 논문의 프로젝트 페이지 | 없음 | — | 논문에 "Video: YouTube"만 적혀 있고 URL 없음. 웹 검색 결과 없음 | 없음 |
| 이 논문 영상 | YouTube (URL 미기재) | — | 확인 불가 | 링크하지 않음 |
| 선행 CON 공식 코드 (NeurIPS 2024, Stölzle & Della Santina) | `github.com/tud-phi/uncovering-iss-coupled-oscillator-networks-from-pixels` → 현재 `phi-lab-research/uncovering-iss-coupled-oscillator-networks-from-pixels` (재지정) | MIT | 별 9개, 최근 push 2026-11-08 (gh API 메타데이터) | 복제하지 않음. FBCON의 전신이므로 필요하면 나중에 정적으로 대조 |

결론: 이 논문에는 공식 코드가 없습니다. 따라서 `repo/`를 만들지 않았고, 코드 대조 문서 대신 주장 대조 문서(`04_claim_verification.md`)를 썼습니다.

## 2. 선행연구 계보

논문의 참고문헌(14쪽)과 본문을 따라 정리했습니다. 출처는 괄호 안의 저자·연도입니다.

**(a) 구조를 가진 학습 동역학**

- **Neural ODE**(Chen et al., 2019): NODE 기준선. 구조 없이 MLP로 ẍ를 예측합니다. 논문 표 1–2의 첫 열.
- **Lagrangian neural networks**(Cranmer et al., 2020): LNN 기준선. M(x)와 U(x)를 따로 학습하고 자동미분으로 동역학을 얻습니다. 논문은 이 방식의 2차 미분 비용을 지적합니다(1절).
- **Input-to-state stable CON**(Stölzle & Della Santina, NeurIPS 2024): 결합 진동자 네트워크 CON. 잠재 퍼텐셜이 강볼록이라 닫힌 형태의 제어를 줍니다. Φ-CON의 직접 전신입니다. 논문은 두 한계를 듭니다. (i) 강볼록 퍼텐셜은 떠 있는 기준점(평행이동 자유도)을 표현하지 못합니다. (ii) 관측과 입력 사상을 따로 학습해서 가상일 원리 아래 물리 힘과 잠재 힘이 일치하지 않습니다(§1, 세 번째 문단).
- **Riemannian reduced-order Lagrangian**(Friedl et al., ICLR 2025): DLO 동역학의 축소 차수 Lagrangian 학습. 논문은 이 연구와 Pierallini et al.을 따라 NODE와 LNN을 비교 대상으로 골랐다고 적습니다(§5.2). Friedl 방식 자체는 비교하지 않았습니다.
- **PINN 케이블 동역학**(Pierallini et al., RA-L 2026): 카테시안 측정으로 케이블 동역학을 추정하는 PINN. LNN 계열과 같이 언급됩니다.

**(b) DLO 학습 모형**

- LSTM(Yang et al., 2022), GNN(Wang et al., 2022; Chu et al., 2024), RSSM(RopeDreamer, Missal et al., 2026, 수백만 파라미터). 논문은 이들이 위치나 속도 명령을 입력으로 쓰므로 "매니퓰레이터가 반력과 무관하게 끝점을 움직일 수 있다"고 가정한다고 비판합니다(§1).
- 준정적 모형과 야코비안 방식(Bretl & McCarthy, 2014; Yu et al., 2025; Shetab-Bushehri et al., 2022; Cao et al., 2024). 동역학 대신 기하나 운동학을 씁니다.

**(c) 제어와 안정성**

- 소프트 로봇 모형 기반 제어: Della Santina et al. (2017, 2023), Borja et al. (2022). 피드백과 피드포워드를 나누는 틀과 에너지 기반 형상 조절을 제공합니다. Φ-CON의 제어식(식 9)은 이 틀을 따릅니다.
- 입력 분리 좌표 변환(Pustina et al., 2022, 2024): Prop. 2의 증명이 Pustina et al. (2024, Prop. 2, 부록 A)을 인용합니다.
- 라그랑주 신경망의 안정성 한계: Müller (2023), Liu et al. (2024). 구조만으로 닫힌 루프 안정성이 나오지 않는다는 근거로 인용합니다.
- 물리 구조 학습 전반: Stölzle (2025, 박사 학위 논문).

**(d) 신체 자원과 데이터 도구**

- Discrete elastic rod(Bergou et al., 2008)와 MuJoCo 플러그인(Korner et al., 2021). 시뮬레이션 기반.
- PyRoki(Kim et al., IROS 2025). 운동학적으로 실행 불가능한 궤적을 거르는 데 씁니다(A.1).

**논문의 새 점.** 세 가지를 합쳤습니다. (1) 떠 있는 기준점을 허용하는 FBCON, (2) 가상일 원리로 힘까지 같이 바꾸는 학습 가능 아핀 Φ, (3) 이 구조에서 나오는 닫힌 형태의 PD + 정적 피드포워드 제어. 개별 요소는 모두 선행연구에 있습니다.

