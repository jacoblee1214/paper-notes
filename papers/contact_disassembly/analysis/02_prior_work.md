# 02. 선행연구, 가까운 대안, 저장소 대조

## 1. 계보

- **촉각 표현의 두 갈래:** 고해상도 변형 영상(GelSight, TacRGB)과 압축된 힘장(TacFF, 광흐름·깊이 기반 전단·법선 격자). 논문은 두 표현을 같은 정책에서 맞붙인다(Sec. I, [34][35]).
- **정책 틀:** Diffusion Policy [5]. 1D U-Net 확산, 관측 조건부. 논문은 이를 그대로 쓰고 관측 인코더만 바꿨다(Sec. II-C).
- **분해 과제:** RL 기반 비틀어 당기기 [12], 언어 유도 분해 [15][16]. 촉각 분해 정책은 드물다.
- **시각 중심 정책의 한계:** 시각 인코더 [17–20], 대규모 시연 [21][22] 덕에 시각만으로도 충분해 보이지만 접촉 전이를 놓친다는 주장([23–25]).

## 2. 가까운 대안 (이 저장소의 다른 분석과 연결)

| 연구 | 같은 질문 | 이 논문과의 차이 | 이 저장소 |
|---|---|---|---|
| ResTacVLA (arXiv:2607.03387) | 촉각을 VLA에 넣을 때 시각이 촉각을 덮는다(modality collapse) | 잔차 VQ 토큰, π0.5 기반. 촉각 표현 비교는 없다 | `restacvla/` |
| GraspSense (arXiv:2604.05697) | 힘 지도로 파지 재정렬 | 파지 선택 단계, 학습 정책 아님 | `graspsense/` |
| GPT-as-Tactile-Policy (arXiv:2610.28457) | 촉각 이해를 범용 모델로 | 고정 모델 + 텍스트 기술. 표현 비교 아님 | `gpt6_astra_tactile/` |
| Beyond Implicit Force (arXiv:2607.14578) | 힘 정보를 관절 오차 δ로 대신할 수 있나 | 원격조작 토크·δ 대조. 촉각 센서 없음 | `beyond_implicit_force/` |
| KineFuse (arXiv:2607.14842) | 평탄한 융합이 시각 특징을 억누른다(병렬 분석 중, 본문은 읽지 않았다) | 본 논문의 결합 결과와 같은 방향의 주장으로 보이나, 확인은 해당 분석에 맡긴다 | 다른 폴더(병렬 작업) |

"구조화된 융합이 필요하다"는 결론은 KineFuse와 ResTacVLA(잔차 표현)가 같은 방향이라 흥미롭다. 다만 이 논문 안에서는 구조화된 융합을 시험하지 않았으므로 **제안만 있고 증거는 없다**(`01` §5).

## 3. 구성 요소별 출처

| 구성 요소 | 출처 | 공개 여부 |
|---|---|---|
| Diffusion Policy | Chi et al. [5], real-stanford/diffusion_policy (MIT) | 공개 |
| ResNet-18 인코더 | 표준 구현 | 공개 |
| IsaacGym + TacSL 촉각 렌더링 | Makoviychuk [36], Akinola [37] | 공개(IsaacGym은 NVIDIA 설치 필요) |
| GelSight R1.5 | Wang, She, Romero, Adelson [38] | 하드웨어 |
| DDIM | Song et al. [39] | 공개 |
| ManiFeel 기반 시뮬 코드 | purdue-mars/manifeel-isaacgymenvs (NOASSERTION) | 공개(라이선스 명시 없음) |

## 4. GitHub와 관련 저장소 (2026-11-09 확인)

| 저장소 | 무엇인가 | 공식? | 라이선스 | 별 | 최근 커밋 (해시) |
|---|---|---|---|---|---|
| AdeeshDesai/CONTACT | 논문 공식 코드(시뮬 학습·평가, 과제 자산). 프로젝트 페이지의 "Code" 링크 | **공식**(저자 1저자 계정) | MIT | 7 | 2026-07-11, `d0ad7f5` (main) |
| purdue-mars/manifeel-isaacgymenvs | IsaacGymEnvs + TacSL 과제 기반(install.sh가 클론) | 의존(구성요소) | NOASSERTION | 1 | 2026-02-25, `1a61da6` |
| real-stanford/diffusion_policy | 정책 기반 | 의존(구성요소) | MIT | 4718 | 2024-12-24, `5ba07ac` |
| iakinola23/tacsl | TacSL 원본 | 의존(구성요소) | 없음 | 3 | 2025-07-11, `ece40d2` |
| gelsightinc/gsrobotics | GelSight 센서 SDK | 하드웨어 | GPL-3.0 | 206 | 2025-06-25, `321d6a2` |

- 프로젝트 페이지 링크: PDF, arXiv, Video, **Code**(AdeeshDesai/CONTACT), CAD 모델(Drive), 데이터(Drive 폴더)(`paper/project_page.html`). 데이터와 CAD는 Drive 링크이며 이 분석에서 받지 않았다.
- `gh search repos "CONTACT tactile disassembly"`: 결과 없음. 다른 구현은 찾지 못했다.
- install.sh는 `real-stanford/diffusion_policy`를 **고정 커밋 없이** 클론한다. 재현할 때 버전이 달라질 수 있다.

## 5. v1 → v2 차이 (텍스트 대조, `.scratch/v1.pdf` 9쪽 vs `paper/paper.pdf` 8쪽)

- **상태:** v1 abs 주석은 "Submitted to IROS 2026"이고, v2는 "Accepted to IROS 2026"이다(abs 페이지 확인).
- **수치:** 표 I–V와 Fig. 1의 숫자 집합이 v1과 v2에서 같다(추출 텍스트 대조). 결과 변경은 없다.
- **독립성 문구:** v1 "Since each configuration is evaluated over 1500 **independent** rollouts"가 v2에서 "1500 rollouts"로 바뀌었다(`03` §2.3 참조). 저자가 독립성 주장을 뺐다는 뜻이며, 본 분석의 우려와 맞닿는다.
- **S1 문장:** v1은 "(81.7% for Vision Only)"를 적었고 v2는 뺐다.
- **S4 문장:** v2에서 S4 설명 단락 위치가 바뀌었다(본문 구성 변화, 수치 변화 없음).
- **문체 변화:** "high-resolution TacRGB" → "TacRGB", "naturally exhibit" → "exhibit"처럼 주장을 약하게 쓴 곳이 있다.
- **참고문헌:** v2에서 RL 관련 참조 [13][14]를 줄이고 번호가 한 칸씩 밀렸다. v1 [15]의 인용 위치가 바뀌었다.
- **프로젝트 페이지:** 시뮬 과제 설명(S1–S5 목록)은 논문과 같다. 영상 목록(R1_cam, R2_tac 등)은 `paper/project_page.html`에 있으나 본 분석에서 재생하지 않았다.

## 6. 우리 연구와의 관계

이 논문은 "촉각 표현 선택"을 실험으로 다룬 드문 비교다. 우리 힘·촉각 정책 설계에서 확인할 것: (a) 힘장을 쓸 때 시뮬과 실물의 정보 차이, (b) 결합을 쓸 때 차원 대 데이터 수, (c) 시드·초기조건을 어떻게 세는지.
