# 01. 논문 분석: VolumeDP (arXiv:2603.17720v2)

- **제목:** VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning
- **저자:** Tianxing Zhou, Feiyang Xue, Zhangchen Ye (equal contribution), Tianyuan Yuan, Hang Zhao, Tao Jiang (Tsinghua IIIS 및 다른 기관 소속)
- **arXiv:** v1 2026-03-18, v2 2026-07-01 (cs.RO), 8쪽. 저널 comment: "Accepted to IROS 2026"
- **라이선스:** arXiv nonexclusive-distrib 1.0 (CC BY 아님). PDF 1쪽에는 arXiv 식별 스탬프만 있고 IEEE 저작권 문구는 없습니다(`pdftotext -f 1 -l 1`로 확인).
- **원문:** `paper/paper.pdf` (v2, 4.3 MB, 25 MB 미만이라 gitignore하지 않음). v1은 `.scratch/v1.pdf`에만 보관.

## 1. v1 → v2 차이

단어 단위 diff(`difflib`, 6,525 대 6,526 단어)에서 본문, 표, 수치 차이는 **없습니다**. 차이는 네 곳입니다.

| 위치 | v1 | v2 |
|---|---|---|
| arXiv 스탬프 | v1, 18 Mar 2026 | v2, 1 Jul 2026 |
| 초록 끝 | 코드 공개 시점 문구("will be released") | "Code and videos are available on the project page: yzc0731.github.io/VolumeDP" |
| 1쪽 각주 | "These authors contributed equally" | "Equal Contribution." |
| 1쪽 연락처 | (v1과 동일) | (v2와 동일) |

따라서 v2의 실질 변화는 **공개 코드에 대한 문장이 바뀐 것**입니다. 그런데 프로젝트 페이지에는 "Code (Coming soon)" 버튼만 있고(§5 참고), v2 초록의 약속과 맞지 않습니다.

## 2. 문제와 주장

- **문제(§I):** 2D 이미지 특징을 3D 행동으로 바꾸는 과정의 2D–3D 불일치가 공간 추론을 해친다고 봅니다. 깊이 센서가 있는 포인트 클라우드(DP3 류)는 비용이 크므로, RGB만 쓰면서 3D 표현을 만드는 것이 목표입니다.
- **핵심 주장:** (i) 이미지 특징을 복셀 볼륨으로 올리면 공간 추론이 좋아진다, (ii) 학습된 가중치로 200개 공간 토큰만 골라도 충분하다, (iii) 토큰 전체를 조건으로 주는 다중 토큰 디코더가 한 벡터로 압축하는 방식보다 낫다.

## 3. 방법 (식 번호와 절은 원문 기준)

| 모듈 | 내용 | 원문 |
|---|---|---|
| 영상 인코딩 | 현재 프레임과 과거 T 프레임의 공유 백본 특징 F2d ∈ R^{H×W×C} | §III-A |
| 볼륨 공간 | 과제별 큐브 범위 r = [(xmin,ymin,zmin),(xmax,ymax,zmax)], 40×40×40 복셀, 채널 32 | §III-A, §IV-A |
| 복셀 중심 좌표 | x = i·s + xmin 등 (식 1) | 식 1 |
| 카메라 투영 | (u,v,1)ᵀ ∝ K P (x,y,z,1)ᵀ, P는 3×4 외부 행렬, K는 내부 행렬 (식 2) | 식 2 |
| 볼륨–이미지 교차 주의 | 복셀 질의가 투영점 주변의 학습된 오프셋에서 샘플링하는 deformable attention (식 3) | 식 3 |
| 목표 조건 | 텍스트 토큰을 복셀마다 붙여 2C 채널 | §III-B |
| 공간 토큰 | TokenLearner식 softmax 가중치 W ∈ R^{N×X×Y×Z}로 가중합, N = 200 | §III-B |
| 보조 감독 | 말단 위치와 그리퍼 개폐 시점 주변에 BCE 마스크, 근거는 proprio 신호 | §III-B |
| 디코더 | 202개 토큰(공간 200 + 언어 1 + proprio 1), 512차원 MLP 투영 후 adaLN-Zero DiT 블록 | §III-C, §IV-A |
| 학습 목표 | DDPM 노이즈 예측 MSE(DP와 동일), 샘플링 100 단계 | 식 4, §IV-A |

## 4. 실험 설정 (원문에 있는 것만)

- **LIBERO:** 4 suite(Spatial, Object, Goal, Long), 과제당 시연 50개, no-op 제거와 실패 시연 제외. 3 시드.
- **ManiSkill:** 5과제(Poke Cube, Place Sphere, Hook Cube, Stack Pyramid, Insert Peg), 과제당 평균 150개 시연(스크립트 정책). 3 시드.
- **LIBERO-Plus:** LIBERO로만 학습한 뒤 7종 교란(카메라, 로봇 초기 상태, 언어, 조명, 배경, 센서 잡음, 배치)으로 평가.
- **실물:** Galaxea R1 Lite(양팔 6-DoF 2개 + 그리퍼), 과제당 시연 100개, 시험 과제당 20회. 4과제(Placing Bowl, Microwave, Door Opening, Nut onto Screw).
- **입력:** 256×256 RGB, 프레임 2장(t, t−1), 언어 지시, 8D proprio.
- **기준선:** BC(LIBERO는 [32]에서 인용, ManiSkill은 ManiSkill 코드베이스의 BC), DP(ManiSkill, LIBERO), DiT-Block Policy.
- **"All methods are uniformly trained and evaluated with 100 denoising steps."** (§IV-A) 이 문장이 공정성 주장의 근거입니다. 다만 epoch 수, 배치 크기, 학습률, 샘플러, 이미지 전처리를 기준선별로 적지 않았습니다.

## 5. 핵심 수치 (표 번호는 원문 기준, 재계산은 `analysis/volumedp_check.py`)

| 항목 | 값 | 표 | 재계산 |
|---|---|---|---|
| LIBERO 평균 Ours / DP / DiT | 88.8 / 72.4 / 74.0 | Table I | 일치 (88.80, 72.40, 74.03) |
| LIBERO suite별 Ours (Spatial / Object / Goal / Long) | 90.7 / 95.9 / 89.9 / 78.7 | Table I | 평균 일치 |
| ManiSkill 평균 Ours / DP / DiT | 68.9 / 49.6 / 38.9 | Table II | 일치 (68.86, 49.6, 38.88) |
| LIBERO-Plus 평균 Ours / DP / DiT | 53.0 / 31.7 / 31.1 | Table III | **열 단순평균은 53.27 / 31.17 / 31.81** (§6 참고) |
| Ablation (LIBERO-Spatial) | 2D 대체 78.2, random 62.8, all 68.4, GT depth 75.8, w/o 감독 82.2, concat 76.4, conv 80.0, 전체 90.7 | Table IV | 전체 대비 차이 일치 |
| 실물 DP / Ours 과제별 | Bowl 16→17, Microwave 12→15, Door 13→17, Nut 5→12 (각 /20) | Table V | 일치 |
| 실물 합계 | DP 46/80, Ours 61/80 | Table V | 일치 |
| 실물 OOD (공간 / 시점 / 환경) | DP 35 / 0 / 20, Ours 65 / 40 / 45 | Table V | 원문 텍스트 추출로 열 배치 확인 |

## 6. 원문 안의 불일치와 주의점

1. **Table III 평균 불일치.** 열 단순평균은 DP 31.17, DiT 31.81인데 표에는 DP 31.7, DiT 31.1로 적혀 있습니다. 두 값이 서로 뒤바뀐 것처럼 보입니다. 원본 데이터 파일이 없어 가중평균 여부는 확인하지 못했습니다. 텍스트 본문도 31.7과 31.1을 그대로 인용합니다. 차이는 0.5포인트 이하라 결론은 바뀌지 않습니다. 다만 "DP 대비 +21.3"의 기준값이 흔들리는 셈입니다.
2. **Fig. 3과 본문의 과제명 불일치.** 본문은 ManiSkill 과제로 **Hook Cube**를 쓰고, Fig. 3 캡션 그림은 **Pull Cube Tool**을 보입니다(렌더링으로 확인). 공개 코드에는 `PullCubeTool-v1`과 `PokeCube-v1`, `PlaceSphere-v1`, `PegInsertionSide-v1`이 있고, `HookCube`와 `StackPyramid`는 등록되어 있지 않습니다(`04` D10).
3. **DP 행의 출처가 표시되지 않음.** Table I에서 DP에는 별표가 없고, 별표는 DiT-Block에만 있습니다. 같은 표의 BC 행은 [32](IB-BC)의 수치와 소수점까지 일치합니다(BC-Trans 67.83±10.42 → 68.0±1.0 등, `02` §2). 그러나 DP 행의 suite별 값(78.3, 92.5, 68.3, 50.5)은 [32]에 없습니다. 같은 [32]의 부록 C.3.1은 DP를 LIBERO-Long에서 **78.0**(50 epoch, 배치 64, 1e-4, 90M 헤드, 128×128, 필터링 없음)으로 보고합니다. 본 논문의 DP Long 값 50.5와 다릅니다. 웹 검색으로도 해당 수치 조합의 출처를 찾지 못했습니다(`02` §2).
4. **"same recipe" 주장의 근거 부족.** 기준선에 대한 명시는 "uniformly trained"뿐입니다. 특히 LIBERO의 DP 행이 재학습인지 인용인지 원문으로 확정할 수 없습니다. 이것이 승격 조건의 핵심입니다(`02`, `03` §8).
5. **컴퓨트 주장 미검증.** §III-B는 토큰 축약으로 계산을 줄인다고 하지만, 파라미터 수, FLOPs, 지연, 학습 시간을 보고하지 않았습니다. 볼륨 40³×32 채널은 약 2.05M 값을 한 장면에서 만듭니다.
6. **Ablation 범위.** 표 IV는 LIBERO-Spatial 한 suite에서만, 시드와 오차 없이 보고했습니다.
7. **LIBERO ± 정의 없음.** ± 값이 SD인지 SE인지, 3 시드 기준인지 밝히지 않았습니다.
8. **GT depth 대조(표 IV e)의 해석.** "GT depth"는 복셀 선택을 점 개수 상위 N으로 하는 휴리스틱입니다. 깊이 센서를 쓰는 기존 3D 방법(DP3 류)과 직접 비교한 것이 아닙니다.
9. **카메라 보정 필요.** 식 2는 외부·내부 행렬을 요구합니다. "RGB only"이지만 보정과 과제별 작업 공간 범위 r은 미리 필요합니다.

## 7. 추가로 확인할 질문

- LIBERO DP 행을 같은 학습 조건으로 다시 학습했는가? (논문에서 확인 불가, 코드도 없음)
- DP와 제안 방법의 학습 epoch, 배치, 샘플러가 같은가?
- 실물 20회 시험의 초기 배치 분포와 시험 순서(무작위화 여부)는 무엇인가?
- 복셀 점유를 만드는 이미지 특징 백본은 DP의 ResNet-18인가, 다른 것인가? 원문에는 "shared backbone"이라고만 적혀 있습니다.
