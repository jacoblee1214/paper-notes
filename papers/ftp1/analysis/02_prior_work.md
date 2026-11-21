# 02. 선행연구, 코드 저장소, 관련 분석

## 1. 공식 코드, 프로젝트 페이지, 관련 저장소

| 이름 | 링크 | 관계 | 라이선스 | 최근 push (UTC) | 별 | 상태 |
|---|---|---|---|---|---|---|
| FTP-1 공식 코드 | [michaelyuancb/ftp1-policy](https://github.com/michaelyuancb/ftp1-policy) | 논문 공식 구현 (openpi 기반, UniVTAC 예제, 사전학습 예제 포함) | Apache-2.0 (LICENSE 파일과 GitHub API 일치) | 2026-07-15 (마지막 커밋 `89fa681`, 2026-07-15) | 150 | 얕은 복제 `repo/`, 정적 읽기만 (`04` 참조) |
| 프로젝트 페이지 | [ftp1-policy.github.io](https://ftp1-policy.github.io/) | arXiv 초록에 적힌 공식 페이지. 논문 v4 로고와 공개 링크 모음 | — | — | — | 확인 |
| 사전학습 체크포인트 | [HF MJJJJ1064/ftp1_v0426_50kstep](https://huggingface.co/MJJJJ1064/ftp1_v0426_50kstep), ModelScope 미러 | 공식 저장소 README가 가리키는 FTP-1 사전학습 50k 체크포인트 | 저장소 README에 명시 없음 (미확인) | — | — | 받지 않음 |
| 파인튜닝 체크포인트 | HF `MJJJJ1064/ftp1_univtac_finetune` | UniVTAC 파인튜닝 체크포인트 | 미확인 | — | — | 받지 않음 |
| FTP-1 데이터셋 | HF `MJJJJ1064/FTP-1-Dataset`, [ModelScope](https://www.modelscope.cn/datasets/Eureka1064/FTP-1-Dataset) | 사전학습 데이터 (논문 Tab. 8) | 미확인 | — | — | 받지 않음 (용량 미확인) |
| openpi | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 기반 코드베이스 (π0.5 PyTorch 포팅) | Apache-2.0 | 2026-08-24 | 14,720 | 링크만 확인 (미복제) |
| UniVTAC | [univtac/UniVTAC](https://github.com/univtac/UniVTAC) | 시뮬레이션 벤치마크 (논문 표 1). 공식 저장소 `UniVTAC/` 폴더에 포함 | Apache-2.0 | 2026-09-07 | 196 | 공식 FTP-1 저장소 안의 복사본을 정적으로 확인 |
| T3 인코더 | README 링크 `alanzjl/t3` 계열 (정확한 주소 미확인) | 이미지형 촉각 인코더 초기값 (논문 [18]) | 미확인 | — | — | 미확인 |

- 저장소 README는 SwanLab, ModelScope, 피슈(Feishu) 그룹 등 외부 서비스를 소개합니다. 이 분석은 그 중 어떤 것도 실행하지 않았습니다.
- 공식 저장소 `AGENTS.md`와 `src/openpi/CLAUDE.md`는 해당 저장소의 개발 지시서입니다. 이 분석에서는 데이터로만 읽었고 따르지 않았습니다.

## 2. 선행연구 계보

논문 App. A(Tab. 5–6)가 정리한 범위를 따릅니다. 여기서는 FTP-1과 가까운 것만 적습니다.


**(b) 촉각 정책 (센서별).** Tactile-VLA [9], OmniVTLA [6], OmniVTA [8], VTDexManip [7], ViTaCFormer [10], FreeTacMan [33], UniVTAC-ACT [27], ViTaL [28]. 논문은 이들을 "특정 센서, 과제, 로봇에 묶인" 방법으로 봅니다.

**(c) 촉각 표현 사전학습.** T3 [18], Sparsh [31], AnyTouch [32], FEATS [84], FeelAnyForce [85]. 여기서 FTP-1이 쓰는 인코더와 프로빙 기준이 나옵니다(T3 인코더, Sparsh와 FeelAnyForce 참조).

**(d) 데이터 출처.** DexUMI [93] (FTP-1 데이터셋에 DexumiInspire와 DexumiXHand로 포함), UMI [86], RH20T [94], Unit [96], RDP [65], exUMI [72], ViTaMIn [70], OpenTouch [91], HumanoidEveryday [92], MotionTrans [47].

**(e) 행동 공간.** UniDex의 FAAS [13]. FTP-1은 손 관절을 FAAS 슬롯으로 맞춥니다.

### 차별점 (논문 Tab. 5–6 기준)

| 항목 | 기존 촉각 정책 (예: Tactile-VLA, OmniVTLA, OmniVTA) | FTP-1 |
|---|---|---|
| 센서 범위 | 보고된 1–4종 또는 NR (Tab. 5) | 학습 21종, 평가 4종 |
| 촉각 입력 주입 | VLM 전문가에 어댑터로 주입 | 독립 촉각 전문가 (블록 어텐션) |
| 학습 안 된 센서 평가 | 보고 없음 또는 NR | 두 종 (Xense 이미지, Contactile 배열) |
| 행동 생성 | 대개 로봇 설정 1–2개 (Tab. 6) | 공유 행동 정책, 평가 설정 5개 (손과 그리퍼 공통) |

논문 기준으로는 "설계 차이"가 맞습니다. 다만 같은 설정 안에서 촉각 입력을 제거한 비교(표 2의 π0.5)는 FTP-1 구조의 이득을 완전히 분리하지 못합니다. 그 점은 `03` C5와 C7에 나옵니다.

## 4. 같은 벤치마크, 다른 결과

UniVTAC 시뮬레이션은 FTP-1 표 1, TacInno, TipDex-VTLA, CoRA-VTLA, GPT-as-Tactile-Policy가 함께 씁니다. 각 논문이 쓴 기준선과 과제 선택이 다르므로, 이 분석에서는 표 값을 서로 옮겨 비교하지 않았습니다. 같은 과제 이름이 나오더라도 설정(데모 수, 시드, 기준선 구현)이 같다고 가정하지 않습니다.

## 5. 찾지 못한 것

- 공식 저장소 README와 논문의 체크포인트 이름 대응(`ftp1_v0426_50kstep`이 논문의 어떤 표 행과 대응하는지)은 확인하지 못했습니다.
- T3 인코더 저장소 주소는 이 분석에서 확인하지 않았습니다.
- 공식 저장소의 issue와 Feishu 그룹 내용은 확인하지 않았습니다.
