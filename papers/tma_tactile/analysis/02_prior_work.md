# 02. 선행연구와 관련 자료

## 1. 계보

TMA는 세 줄기를 잇습니다.

1. **적응형 정책 (RMA 계열).** RMA(Kumar et al., arXiv:2107.04034, 2021)는 특권 인코더로 잠재를 학습하고, 이력에서 잠재를 추정하는 적응 모듈을 학습합니다. TMA의 두 단계 구조(오라클 정책 → 지도 회귀 적응 인코더)는 여기서 왔습니다. 손 안 회전에 RMA를 쓴 Qi et al.(CoRL 2023, ref [33])이 가장 가까운 선행입니다.
2. **촉각 sim2real.** 촉각 정책을 시뮬레이션에서 실물로 옮기는 연구들입니다. Qi et al. (CoRL 2023, ref [21], 시각과 촉각 손 안 회전), AnyRotate (CoRL 2024, ref [22], 스칼라 힘 크기), Bi-touch (RA-L 2023, ref [26]), DexTouch (RA-L 2024, ref [24]), TacGNN (RA-L 2023, ref [28]), Robot Synesthesia (ICRA 2024, ref [27]). 논문은 이들이 이산 접촉 신호나 스칼라 힘을 쓴다고 지적하고, 방향을 가진 힘 표현을 주장합니다(Sec. II).
3. **휴머노이드 조작과 문 열기.** 휴머노이드 원격조작과 모방 학습(OmniH2O, TWIST, HOMIE, HumanPlus, OKAMI)과 힘 적응 loco-manipulation(FALCON, ref [20])이 배경입니다. 문 열기에 가까운 것은 다음과 같습니다. Xue et al.(CVPR 2026, ref [5], 픽셀에서 행동까지 문 열기 sim2real), DoorBot(arXiv:2504.09358, ref [6], 모터 전류를 햅틱 피드백으로 쓰는 문 열기), Holladay et al.(arXiv:2208.00319, ref [7], 다단계 힘 조작 계획).

## 2. 가까운 대안과 GitHub 자료

### 2.1 가까운 대안 (논문이 인용한 것 중)

| 연구 | 무엇이 같은가 | 무엇이 다른가 |
|---|---|---|
| Zhao et al., arXiv:2601.02778 (ref [29]) | 조밀한 힘 피드백으로 다지 손의 힘 기반 파지와 조작을 zero-shot 시뮬 sim2real | 적응 인코더 없이 힘 피드백을 직접 씀. TMA의 원뿔 좌표와 이력 적응은 없음 |
| HydroShear, arXiv:2603.00446 (ref [30]) | 촉각 sim2real을 위한 전단 시뮬레이션 | 센서 모델링에 집중. 정책 구조는 다루지 않음 |
| SimShear, arXiv:2508.20561 (ref [31]) | 전단 기반 촉각 서보 sim2real | 촉각 서보 과제. 문 열기 없음 |
| Yang et al., arXiv:2608.15897 (ref [32]) | 촉각 sim2real을 병목 잠재 재구성으로 해결 | 촉각 시뮬을 거치지 않음. 적응 이력 없음 |
| Xue et al., CVPR 2026 (ref [5]) | 휴머노이드 문 열기 sim2real | 시각 픽셀에서 행동까지. 촉각 없음 |

### 2.2 GitHub와 프로젝트 자료 (조사일 2026-11-21)

| 자료 | 주소 | 라이선스 | 스타 | 최근 push | 상태 |
|---|---|---|---|---|---|
| 논문 공식 코드 | 찾지 못함 (arXiv HTML, 프로젝트 페이지, GitHub 검색 "2611.18976", "humanoid-tma", "tactile motor adaptation doorknob" 확인) | - | - | - | 공식 코드 없음 |
| 프로젝트 페이지 저장소 | https://github.com/humanoid-tma/humanoid-tma.github.io (생성 2026-08-13) | 없음 | 0 | 2026-11-19 | 웹사이트 자산(HTML, CSS, JS, 영상, 그림)만 있음. 정책이나 학습 코드 없음. 파일 목록만 확인함(클론하지 않음) |
| 프로젝트 페이지 | http://humanoid-tma.github.io | - | - | - | 영상, 그림, BibTeX (`journal = arXiv preprint`). 코드 링크 없음 |
| mjlab (시뮬레이터, ref [35]) | https://github.com/mujocolab/mjlab | Apache-2.0 | 3,437 | 2026-11-21 | 논문이 쓴 시뮬레이터. Isaac Lab API와 MuJoCo-Warp 기반 |
| 로코모션 컨트롤러 HOMIE (ref [12]) | 저장소 주소 확인 실패 | - | - | - | 미확인 |

**결론:** 공식 코드는 없습니다. 그래서 `analysis/04_claim_verification.md`로 주장을 검증합니다. 코드 저장소를 clone하지 않았습니다(`repo/` 폴더 없음).

## 4. 구성 요소별 출처 정리

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 잠재 적응과 두 단계 학습 | RMA (Kumar et al., 2021, ref [9]) | 촉각 입력과 원뿔 좌표는 TMA의 새 부분 |
| 손 안 회전에서 RMA | Qi et al. (CoRL 2023, ref [33]) | 같은 구조의 손 조작 선행 |
| 원뿔 좌표 힘 표현 | 마찰 원뿔 기하 (Li, Sastry, Murray, 1994, ref [37]) | 식 (8)의 기하 근거 |
| 정책 학습 | PPO (Schulman et al., 2017, ref [38]) | 비대칭 critic 사용 |
| 인코더 구조 | GRU (Cho et al., 2014, ref [39]), TConv (Bai et al., 2018, ref [40]) | Fig. 5에서 TConv 선택 |
| 시뮬레이터 | mjlab (ref [35]), MuJoCo (ref [36]) | 접촉 데이터 접근 |
| 로코모션 컨트롤러 | HOMIE (ref [12]) | 동결 사용 |
| 촉각 sim2real 선행 | ref [21]–[34] | 2.1 표 참조 |
