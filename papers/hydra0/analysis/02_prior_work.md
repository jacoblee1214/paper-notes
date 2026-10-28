# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다. 각 논문을 별도로 열어 검증하지는 않았습니다.

## 1. 계보

```
[행동 조건 world model]  Dreamer [20], DINO-WM [58]  →  비디오 생성형: Cosmos 2.5/3 [36, 35], IWS [48], [19]
        │  문제: 행동이 관절·EE 좌표라 embodiment마다 매핑을 새로 배워야 함
        ▼
[궤적·움직임 조건 비디오 생성]  Motion Prompting [17] → ATI [45](Gaussian 전파) → Wan-Move [13](첫 프레임 특징을 궤적 따라 복제)
        │  문제: 사용자가 그린 궤적일 뿐, 실행 가능한 로봇 명령과 연결되지 않음
        ▼
★ Hydra-0  로봇 기하와 보정으로 명령 → 픽셀 궤적(kinematically grounded) + 다중 embodiment mid-training + 역방향 모드
```

## 2. 가까운 대안 (논문 Related Work 기준)

| 방법 | 2D 조건 | Hydra-0과의 차이 |
|---|---|---|
| MIMIC [8] | 참조 기반 의미 마스크 | 점 궤적이 아님 |
| Mask2IV [28] | 행위자·물체 마스크 궤적 | 마스크 단위 |
| BridgeV2W [10] | 명령으로 렌더링한 embodiment 마스크 | 로봇 렌더 마스크 |
| OSCAR [49] | 2D 운동학 스켈레톤 렌더 | 스켈레톤 |
| ContactFlow [4] arXiv:2607.26579 | 접촉 flow 조건 | 접촉 중심 |
| 기타 | kinematic pointmap 조건, optical-flow WAM, 로봇 렌더링, flow 중심 생성 계획, 궤적 조건 교차 embodiment 기술 전이 [51, 9, 27, 16, 38] | — |

**차별점(저자 주장):** 가시성을 고려한 로봇 표면 점 궤적, 다중 embodiment 학습, 정방향과 역방향의 공유 인터페이스, 그리고 메타데이터 없는 영상에서도 학습할 수 있는 track 복원입니다.

## 3. 구성 요소별 출처

| 역할 | 출처 |
|---|---|
| 비디오 백본 | Cosmos 2.5 [36], Wan2.2 [44], LightX2V 4-step 체크포인트 [1] |
| 조건 주입 | ATI [45], Wan-Move [13] |
| 자기회귀 변환 | LongLive-2.0 [11], CausVid [55] 등 [55, 59] |
| 증류 | DMD [54], DMD2 [53] |
| 추적과 분할 | AllTracker [21], SAM 3 [7] |
| 미세조정 | LoRA [24] |
| 평가 | RoboLab [52] arXiv:2604.09860 (시뮬레이션 벤치마크), Gemma 4 31B VLM 판정 [40] |
| 데이터 | DROID [26], ABC-130k [3] arXiv:2606.27375, MolmoAct2 [14], EgoDex [23], Deform360 [30], X-VLA Soft-Fold [57] |

## 4. 이 저장소에서 분석한 다른 논문과의 관계
- **Do as I Do** (`../do_as_i_do`): 사람 영상을 3D로 복원한 뒤 물리 리타겟팅으로 로봇 궤적을 만듭니다. Hydra-0은 같은 사람 영상을 **2D flow 그대로** 학습 신호로 씁니다. 두 논문은 "사람 영상 → 로봇" 문제에 대한 3D 명시적 경로와 2D 암묵적 경로라는 정반대의 선택입니다.
- **DexScrew** (`../dexscrew`): 접촉과 촉각은 시뮬레이션이 어렵다는 같은 문제의식에서 출발합니다. DexScrew는 실제 데이터로 우회하고, Hydra-0은 학습된 비디오 모델로 우회합니다. Hydra-0의 한계 절이 "촉각·힘 조건이 다음 단계"라고 적은 것도 이 지점입니다.
