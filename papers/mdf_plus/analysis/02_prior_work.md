# 02. 선행연구와 관련 저장소

이 문서는 MDF++의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(논문 원문 PDF의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| MDF++의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 시간·모달리티별 노이즈 수준으로 마스킹 표현 | Diffusion Forcing [49] (Chen 2024), MDF [10] (Huang, Hou, Berenson 2025) | MDF++의 바탕입니다. MDF는 프로즌 점군 VAE를 씁니다 |
| 구조화된 2D 노이즈 샘플링 (정책, 세계-행동, 순·역 동역학, 무작위 마스킹) | 논문 자체 기여(§4.1) | MDF는 무작위 K를 씁니다. 이 조합의 효과는 표 1의 MDF(adapted) 대 MDF++(policy)와 MDF++로 나눠 보입니다 |
| 자기 복원 손실로 인코더가 무너지는 문제와 해결 | REPA-E [54] (Leng 2025) | 인코더를 확산 손실로 끝까지 학습하면 무너진다는 관찰의 출처입니다. 논문은 "자기 스트림 그래디언트 차단"으로 바꿉니다 |
| 잠재 공간 정규화 기준선 | SIGReg [58] (LeJEPA, Balestriero·LeCun 2025), LeWorldModel [15] (Maes 2026) | MDF++(SIGReg) 기준선입니다. 저자는 정규화가 부족하다고 결론합니다 |
| 조작용 인코더 대안 (촉각) | Sparsh [45] (Higuera 2024) | MDF++(frozen-Sparsh) 기준선입니다 |
| 의미 중심 인코더 대안 | ViT [55], ViT 학습 방법 [59] | MDF++(frozen-ViT) 기준선이고 인코더 초기값으로도 쓰입니다 |
| 재구성 기반 WAM | UWM [9] (Zhu 2025), UVA [2] (Li 2025), SDXL VAE [20] | UWM은 실험 기준선입니다. 저자는 픽셀 수준 재구성이 용량을 낭비한다고 봅니다 |
| 세계-행동 모델 (WAM) 계열 | GR-2 [1], Dual-stream diffusion [3], VPP [5], Cosmos Policy [6], Video generators are robot policies [7], mimic-video [8], WorldVLA [53] | 관측 예측과 행동을 한 모델에서 씁니다. MDF++가 속한 계열입니다 |
| 보조 동역학 목표가 표현을 돕는다는 근거 | DynaMo [50], 역동역학 사전학습 [51], PAD [52] | 순·역 동역학 조건부를 쓰는 근거입니다 |
| 촉각·힘 기반 정책 | 3D-ViTac [32], VITAL [33], VITamin [34], 촉각 Transformer [40], DP-tactile(Touch in the wild) [36], TacDiffusion [31], Forge [29] | 기준선 DP w/ tactile [36]과 시뮬 환경 Forge [29]의 출처입니다 |
| 촉각 표현 학습 | UniTac [35], Sparsh [45], Transferable tactile [46], 마스크드 촉각 [44] | 촉각 인코더 대안입니다. 논문은 이 계열이 제어 목적 학습을 하지 않는다고 봅니다 |
| 시뮬 과제 | Isaac Lab [56], ManiFeel [57] | PegInsert, GearMesh, NutThread(Isaac Lab FACTORY)와 BallSorting, ObjectSearch(ManiFeel)입니다 |
| 실로봇 데이터 수집 | UMI [61], 원격 조작 어려움 [60] | 잔차 보정이 필요한 근거입니다 |
| 잠재 확산 학습 | Unified Latents [62] (Heek 2026) | 저자가 향후 과제로 언급합니다 |


## 2. 코드와 저장소

### 2.1 공식 코드 조사 결과 (2026-11-21)

| 확인 대상 | 방법 | 결과 |
|---|---|---|
| 프로젝트 페이지 jointmdf.github.io | HTML 전체 읽기 (`href` 추출) | GitHub 링크 없음. 논문 PDF, BibTeX, 그림 정적 파일뿐 |
| arXiv abs 2611.09293 | abs 페이지 읽기 | 코드 링크 없음. "view license"는 arXiv 비독점 배포 라이선스 |
| GitHub 저장소 검색 (API) | "MDF++", "visuo-tactile diffusion forcing", "multimodal diffusion forcing", "unified multimodal diffusion forcing", "forceful manipulation diffusion forcing", "visuo-tactile world action model", "tactile world action model diffusion" | 관련 저장소 없음. 이름이 같은 무관한 저장소만 나옴(`mdflib`, `asammdf` 등) |
| 저자 GitHub 계정 | 사용자 저장소 목록 (`zixuanh`) | 2014–2021년의 개인 프로젝트 5개만 있음. MDF 관련 없음 |
| 저자 연구실 조직 RoboticExplorationLab | 조직 저장소 목록 (3쪽, 211개 전부 조회) | MDF 또는 다중 모달 확산 관련 저장소 없음 |
| 선행 MDF (arXiv:2511.04812) | abs 페이지 읽기 | 프로젝트 웹사이트 링크만 있고 코드 링크는 확인하지 못함 |

**결론: 공식 코드를 찾지 못했습니다.** 따라서 `analysis/04_claim_verification.md`로 논문 주장을 검증했습니다. 코드를 찾으면 `repo/`에 얕은 복제(정적 읽기)를 하고 `04_code_verification.md`로 바꿉니다. 저자가 나중에 코드를 공개할 수 있으므로 이 조사는 날짜 기준입니다.

### 2.2 이 분석에서 쓰지 않은 것
- 가중치, 데이터셋, 영상 자료는 받지 않았습니다. 프로젝트 페이지의 영상도 보지 않았습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- 자기 복원 손실로 인코더를 학습하면 무너진다(REPA-E [54]의 관찰). 논문의 핵심 동기입니다. 이 논문 안에서 직접 확인한 증거는 표 2와 표 1의 naive grad 행뿐입니다.
- 보조 동역학 목표는 표현을 돕는다([50–52]). MDF++(policy)와 MDF++의 차이(78.9 대 86.9)로 이 논문의 기여를 나누어 볼 수 있습니다.
