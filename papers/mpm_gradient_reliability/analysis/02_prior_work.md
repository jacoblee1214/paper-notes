# 02. 선행연구와 관계

번호 [n]은 논문의 참고문헌 번호입니다(`paper/paper.txt`의 References). 논문 밖 문헌은 이 문서 작성자의 판단이며, 원문을 열어 확인하지 않은 것은 그렇게 표시했습니다. GitHub 정보는 2026-11-11에 `gh`로 확인했습니다.

## 1. 계보

| 줄기 | 논문 | 이 논문과의 관계 |
|---|---|---|
| MLS-MPM / APIC | Hu et al., ACM TOG 2018 [6] | 이 논문의 수치 기반. 격자 전달과 APIC 속도 표현을 그대로 씀 |
| 미분 가능 MPM 식별 (탄소성) | Yang, Ji, Lai, IJRR 2025 [1] (DPSI) | 첫 번째 테스트베드(고정 corotated + von Mises). CD와 EMD가 다른 파라미터로 수렴한다고 보고 |
| 미분 가능 굴삭 (입상) | Yang, Wei, Lai, Ji, IEEE TRO 2026 [2] (DDBot) | 두 번째 테스트베드(Hencky/StVK + Drucker–Prager). 기울기 폭발·진동, 기울기 클리핑과 라인서치 사용 |
| 미분 가능 시뮬레이션 응용 | Wei et al. TNNLS 2026 [3], Wei et al. IROS 2025 [4], Ye et al. TRO 2026 [5] | 시연 생성, 스킬 최적화, 형상 설계. 이 논문이 "같은 수치 연산"이 여기에도 있다고 주장하는 대상 |
| 재현 가능한 부동소수점 합 | Demmel & Nguyen, ARITH 2013 [7], IEEE TC 2015 [8] | 결정론 누적(고정소수점 정수 합)의 이론 근거 |
| 접촉 기울기 | Werling et al. RSS 2021 [9], Zhong et al. L4DC 2023 [10], Liu et al. IROS 2024 [11] | 접촉과 이벤트 시점이 기울기를 바꾼다는 선행 관찰 |

논문 밖 배경(작성자 판단, 원문 미확인): 미분 가능 시뮬레이션에서 긴 롤아웃의 기울기가 혼돈이나 접촉 불연속 때문에 나빠진다는 주장은 여러 연구에서 다뤄졌습니다. 이 분석은 개별 문헌을 열어 확인하지 않았으므로 이름을 특정하지 않습니다.

## 2. 가까운 대안

### 2.1 논문이 인용했지만 비교하지 않은 것

- [9]–[11]: 접촉 기울기의 개선(접촉 힘의 예측 모형, 이벤트 타이밍). 이 논문은 접촉 모형을 고정했으므로 비교하지 않았습니다. 접촉 쪽을 바꾸면 "접촉 불연속" 원인이 어떻게 바뀌는지는 답하지 못합니다.
- [7], [8]: 재현 가능 합의 다른 구현(예: 2단계 블록 합, 정렬 합)과 성능 비교가 없습니다. 고정소수점 방법만 보입니다.

### 2.2 논문이 인용하지 않았지만 가까운 대안(작성자 판단)

- **AD/FD 검사를 위한 복소수 단계(complex-step) 또는 전진 접선 AD.** 절단 오차가 없고 뺄셈 상쇄도 없어서, FD 대신 기준으로 쓸 수 있습니다(단, 해석 함수에 한함). 혼돈 롤아웃에서는 접선 AD 자체가 e^{λT}로 커지므로 기준으로는 Decimal이나 장정밀도 접선이 더 안전합니다. toy에서 그렇게 했습니다(03 §1).
- **안정적인 SVD 미분 식.** 특이값 차이에 대한 divided difference, 또는 간격이 임계 아래일 때 Taylor 전개. 게이지 불변 손실에 대해서는 A와 B가 반대 부호라는 성질로 F 항이 상쇄되어 1/g 특이성이 사라집니다(toy `analysis/svd_vjp_toy.py`에서 확인).
- **결정론 누적의 GPU 대안.** 원자 덧셈 대신 정렬 후 세그먼트 합, 또는 CSR 행 기준 gather (논문 부록 B-C가 이미 쓰는 방식). 대가는 메모리와 추가 정렬 비용입니다.

## 3. 이 저장소의 다른 분석과의 관계

| 폴더 | 무엇을 다루나 | 이 논문과 맞닿는 곳 |
|---|---|---|
| `environmental_pdes/` | 이류–확산 PDE의 격자·시간 간격과 1차 풍상 수치 확산 | 수치 방법이 측정 모형의 유효 물리를 바꾼다는 점. 이 논문의 "수치 설정이 기울기를 바꾼다"와 같은 질문 구조. 단위 판정(유효 확산 식)의 검산 방식 |
| `robosimstudio/` | Newton 기반 시뮬레이션 도구, 사람과 코딩 에이전트가 solver와 물성을 고쳐 Real2Sim | 시뮬레이션 식별 도구의 기울기 검증 자리. 이 논문의 AD/FD 프로토콜을 넣을 수 있는 위치 |
| `reactive_grasp/` | 접촉력 분배 QP와 접점 계산 | 접촉 모형의 불연속(접촉 진입·이탈) 문제. 이 논문이 고정한 접촉 모형을 바꾸면 생길 문제 |

## 4. GitHub 저장소 (2026-11-11 확인)

| 저장소 | 무엇인가 | 관련성 | ★ | 마지막 push | 라이선스 |
|---|---|---|---|---|---|
| geoelements/diffmpm | Differentiable Material Point Method (Python) | 미분 가능 MPM 구현 예. 이 논문의 코드 아님. 역전파와 SVD 처리 방식 비교용 | 48 | 2023-09-01 | 없음(명시 안 됨) |
| Chayoso/A-Differentiable-Material-Point-Method-for-Shape-Morphing | 설명 없음 (C++) | 형상 모핑용 미분 가능 MPM으로 보이나 논문 연결은 확인 못 함 | 2 | 2025-03-31 | 없음 |
| arxiv-digest (mktk1117) | arXiv 요약 모음의 `details/2609.34666/` 메타데이터 | 코드 아님. 그림 파일명 등 메타데이터 미러 | — | — | — |
| Assaro/DD_Bot | Discord 도커 봇 | 이름만 같음. DDBot과 무관하여 제외 | 45 | 2024-06-30 | GPL-3.0 |

- `gh search repos`로 "differentiable MPM taichi", "DPSI differentiable physics system identification", "DDBot digging robot granular", "differentiable physics robotic material manipulation", "differentiable digging granular taichi"를 검색했고 결과가 0건이었습니다. 코드 검색(`gh search code "2609.34666"`)도 0건입니다.
- 저자 이름으로 된 GitHub 계정은 확인하지 못했습니다(동명 계정은 본인 여부를 확인할 수 없어 제외).
- arXiv abs 페이지에 GitHub, 프로젝트 페이지 링크가 없습니다.
- 결론: **공식 코드를 찾지 못했습니다.** `repo/`는 만들지 않았고, 코드 대조 문서 대신 `04_claim_verification.md`를 썼습니다.

## 5. 이 분석에서 가져오는 것

1. **AD/FD 검사 프로토콜:** 섭동 크기 사다리를 쓰고, 기준은 전진 접선 또는 장정밀도 접선으로 둔다. 불일치가 h에 따라 줄면 FD 탓, 고원이면 AD 또는 불연속 탓으로 본다(`analysis/fd_chaos_toy.py`).
2. **손실 산포 기준:** 섭동 효과와 반복 산포는 같은 정밀도로 비교한다(03 §2).
3. **SVD VJP 점검:** 특이값 간격을 로그로 남긴다. 간격 1e-7 이하에서 F 항의 오차가 커지는지 확인한다(`analysis/svd_vjp_toy.py`).
