# 02. 선행연구, 가까운 대안, 자료와 저장소

## 1. 선행연구 계보

MimicGen은 세 흐름이 만나는 자리에 있습니다. (논문 §2, 부록 E, J의 인용 번호를 따랐습니다.)

1. **사람 원격 조작으로 대량 데이터 모으기.** RoboTurk(Mandlekar 외, 2018)[2], 원격 조작 기반 모방학습(Mandlekar 외, 2020)[24], BC-Z(Jang 외, 2022)[3], RT-1(Brohan 외, 2022)[5]. 데이터는 크지만 사람 시간과 비용이 큽니다. 논문은 RT-1의 1.5년 수집을 예로 듭니다(§1).
2. **재생 기반 모방학습.** 시연 하나를 새 장면에 맞춰 재생합니다. YOTO(Wen 외, RSS 2022)[8], Coarse-to-fine(Johns, ICRA 2021)[9], DOME(Valassakis 외, IROS 2022)[10], Self-replay(Di Palo & Johns, CoRL 2022)[11], Vosylius & Johns[46], Chenu 외[47], Liang 외[48]. 대부분 정책 안에 재생 단계를 넣어 **정책이 열린 고리 재생에 기대는** 구조입니다(부록 J.1). MimicGen은 같은 변환 아이디어를 **데이터 생성기**로만 씁니다. 학습 정책은 재생을 하지 않습니다.
3. **오프라인 데이터 증강.** Mitrano & Berenson(2022)[35]와 픽셀 이동 무작위화[36–39]. 오프라인으로 변형하므로 물리 일관성이 보장되지 않습니다(부록 J.2). MimicGen은 시뮬레이터에서 실행해 얻은 성공 궤적만 씁니다.

시뮬레이션 기반 시연 생성의 선행으로 논문은 사전 프로그램된 시연자[18–22]를 듭니다. MimicGen의 차이는 사람 원천 시연을 쓰고 객체 상대 변환을 쓴다는 점입니다.

## 2. 가까운 대안과 구분되는 점

| 방법 | 데이터 형태 | 장면 변화 처리 | MimicGen과 차이 |
|---|---|---|---|
| 원격 조작 확장 (RT-1, BC-Z) | 사람 시연 | 사람이 직접 장면을 바꿈 | 사람 시간이 생성량에 비례 |
| 재생 기반 정책 (YOTO, Self-replay) | 시연 하나 | 객체 자세 정합 후 재생 | 정책이 재생 구조에 묶임. 데이터셋을 만들지 않음 |
| 오프라인 증강 (Mitrano & Berenson) | 기존 데이터 변형 | 이미지 또는 자세를 오프라인 변형 | 물리 일관성 없음 |
| **MimicGen** | 소수 사람 시연 | 객체 상대 세그먼트 변환 후 시뮬에서 실행, 성공만 저장 | 정책 구조와 무관, 데이터셋 생성 |
| DexMimicGen (Jiang 외, arXiv:2410.24185) | 양손 다지 손 시연 | MimicGen 계열을 양손 다지 로봇으로 확장 | 논문 이후 후속 연구 (arXiv 확인) |
| SkillMimicGen (Garrett, Mandlekar 외, arXiv:2410.18907) | 시연과 기술 단위 | 기술 단위로 생성 | 논문 이후 후속 연구 (arXiv 확인) |

## 4. GitHub와 자료 (2026-11-22 확인)

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [NVlabs/mimicgen](https://github.com/NVlabs/mimicgen) | 공식 데이터 생성 코드, 시뮬 과제 구현(robosuite 기반 12개 과제 계열), 문서 | NVIDIA Source Code License (LICENSE 파일, 비상업 연구·평가 전용, §3.3). GitHub API는 NOASSERTION | 2026-10-21 | 673 | 로컬 얕은 복제 (커밋 `c0fbff2`, 2026-10-21). 정적 읽기만 |
| [NVlabs/mimicgen_environments](https://github.com/NVlabs/mimicgen_environments) | 프로젝트 페이지 "Code" 링크 | API 조회 시 `NVlabs/mimicgen`으로 리디렉션 | - | - | 별도 복제 없음 |
| [mimicgen.github.io](https://mimicgen.github.io) | 프로젝트 페이지, 문서, 데이터 안내 | - | - | - | 페이지 확인. "Update (July 2024): Data Generation Code has been released" |
| [amandlek/mimicgen_datasets](https://huggingface.co/datasets/amandlek/mimicgen_datasets) (Hugging Face) | 공개 데이터셋 (README: CC-BY 4.0) | CC-BY 4.0 (README 기준) | - | - | 받지 않음 (요청 범위 밖) |
| [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | BC-RNN 학습 프레임워크 (논문 학습 설정의 출처) | MIT | 2026-08-09 | 1608 | 외부 의존성. 저장소에 없음 |
| [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) | 시뮬레이션 프레임워크 (논문 [49]) | GitHub API NOASSERTION (LICENSE 파일 확인 필요) | 2026-11-19 | 2721 | 외부 의존성 |
| [isaac-sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) | Factory 과제 참고 구현 (논문 [51], [52] 계열) | NOASSERTION | 2024-10-26 | 2949 | 공식 MimicGen 저장소에 Factory 과제 구현 없음 |

- 공식 코드 저장소는 2023년 논문 시점 코드가 아닙니다. 2024년 릴리스 이후 코드입니다(README "Latest Updates": v1.0.0 2024-07-09, v1.0.1 2024-09-19). 논문 실험을 그대로 재현하는 코드인지는 `04_code_verification.md`에서 따로 봅니다.
- 데이터셋은 Hugging Face에 있고 README는 CC-BY 4.0이라고 씁니다. 이 분석에서 받지 않았습니다.
