# Robot Learning for Physical AI

> 로봇 러닝의 기초를 학습하고 Physical AI 오픈소스 분석·재현·기여로 연결하는 가짜연구소 13기 프로젝트입니다.

[프로젝트 페이지](https://pseudo-lab.com/projects/326b9bb8-c161-4757-ace1-09da00c852d9) · [가짜연구소](https://pseudo-lab.com/) · [Discord](https://discord.gg/EPurkHVtp2)

## 프로젝트 소개

ETH Zurich의 「Robot Learning: From Fundamentals to Foundation Models」 강의를 따라 로봇 러닝을 공부합니다. 매주 한 강을 다루며, 발표자가 강의 내용을 설명하고 함께 질문하는 방식으로 진행합니다. 과제는 학습 주제와 팀 여건에 맞는 항목을 선택해 실습합니다.

후반부에는 Physical AI 오픈소스를 나누어 조사하고 직접 설치해 봅니다. 재현 과정에서 찾은 문제는 해당 프로젝트에 issue나 PR로 남깁니다. 공통으로 겪는 문제가 확인되면 이를 해결할 작은 오픈소스 MVP를 만듭니다. 구현 범위는 사용할 수 있는 컴퓨팅 자원에 맞춰 정합니다.

시즌 종료 후에는 참여자들과 후속 연구·개발 주제를 정하고 Physical AI 리서치랩 활동을 이어갈 계획입니다.

## 강의 자료

- [ETH Zurich — Robot Learning: From Fundamentals to Foundation Models](https://cvg.ethz.ch/lectures/Robot-Learning/)
- [공식 과제 저장소](https://github.com/mees-robot-learning-course/ethz-course-2026)
- 강의 1강은 OT 전에 자습하고, 정규 모임에서는 2강부터 시작합니다.

## 🎯 Goal

- [ ] 로봇 제어, MDP, imitation learning과 reinforcement learning의 핵심 원리를 이해하고 구현합니다.
- [ ] 강의 과제 중 일부를 골라 Behavioral Cloning, DAgger, PPO 또는 SAC의 동작을 실험합니다.
- [ ] Generative policy, transformer, world model, VLA와 embodied reasoning을 학습합니다.
- [ ] Physical AI 오픈소스를 설치·실행하고 재현성, 사용성, 확장성과 라이선스를 분석합니다.
- [ ] 발견한 문제를 issue, 문서, 테스트 또는 코드 PR로 기여합니다.
- [ ] 반복적으로 확인된 문제를 해결하는 오픈소스 MVP를 설계하고 공개합니다.
- [ ] 시즌 종료 후에도 연구와 개발을 이어갈 Physical AI 리서치랩 기반을 만듭니다.

### 📦 Expected Outcome

- 강의 학습 및 발표 기록
- 선택한 프로그래밍 과제의 구현·실험 기록
- Physical AI 오픈소스 분석·재현 기록과 기여 내역
- 반복되는 미해결 문제가 확인되면 이를 다루는 오픈소스 MVP

## 🏁 Milestones

1. **학습 체계 구축**
   - 공통 개발 환경, 발표 일정, 과제 방식, AI 사용 규정 확정
2. **핵심 알고리즘 구현**
   - Imitation Learning과 Reinforcement Learning 학습
   - 선택한 과제의 일부를 구현하고 실험 결과 공유
3. **Physical AI 연구 주제 선정**
   - Transformer, World Model, VLA와 Embodied Reasoning 학습
   - 참여자별 관심 연구 주제 선정
4. **오픈소스 분석 및 재현**
   - Physical AI 오픈소스 설치·재현
   - 재현성, 사용성, 확장성과 라이선스 분석
5. **신규 오픈소스 MVP 개발 및 공개**
   - 확인된 생태계 문제를 다루는 MVP 개발
   - 컴퓨팅 자원에 맞춰 구현 범위 조정
   - 후속 리서치랩 활동 계획 수립

## 🗺️ Weekly Roadmap

정기 모임은 **매주 화요일 21:00–23:00, 디스코드 Room-CS**에서 진행합니다. 날짜와 미정 주차는 추후 확정합니다. 아래는 프로젝트 페이지에 등록된 계획이며, 주차별 결과물은 아직 미정입니다.

| 주차 | 주요 활동 |
|---|---|
| W01 | 목표, 운영 방식, 발표 순서, 과제 진행법과 AI 사용 규정 공유. 사전 자습한 강의 1강 토론 |
| W02 | 로봇 제어 기초와 Markov Decision Process |
| W03 | Behavioral Cloning, covariate shift와 DAgger |
| W04 | Value function, policy gradient와 exploration |
| W05 | PPO, SAC와 offline·online reinforcement learning |
| W06 | Energy-based model과 diffusion model |
| W07 | Decision Transformer, ACT와 trajectory sequence modeling |
| W08 | 미정 |
| W09 | World model과 planning |
| W10 | Generalist robot policy와 VLA |
| W11 | Embodied reasoning 및 test-time scaling |
| W12 | Physical AI와 robot learning의 최신 연구 방향 및 미해결 문제 토론 |
| W13 | 미정 |
| W14 | 미정 |
| W15 | 미정 |
| W16 | Ending: 학습 및 오픈소스 활동 결과 공유, 회고와 후속 활동 논의 |

## 👥 Team

| 역할 | 이름 | 담당 |
|---|---|---|
| Builder | [박진호](https://github.com/jjinCode) | 프로젝트 기획·리딩 및 운영 |
| Runner | 추후 확정 | 사전 학습, 순환 발표, 과제 수행, 오픈소스 분석·기여 |

프로젝트 정원은 6명입니다. 빌더와 러너는 서로 질문과 피드백을 주고받으며 함께 학습합니다.

## 🧑‍🤝‍🧑 How We Work

- 매주 지정된 강의와 자료를 사전 학습합니다.
- 발표 순서는 참여자끼리 정합니다. 담당자는 강사처럼 개념, 수식, 코드와 선택 이유를 직접 설명하고 질문에 답합니다.
- 과제 전체 완주를 목표로 하지 않습니다. 학습 주제와 팀 여건에 맞는 항목을 골라 수행합니다.
- 코드 리뷰와 오픈소스 조사·개발에 참여하고 시행착오를 기록합니다.
- 팀 성과를 공유하며, 팀원의 자료를 사용할 때는 허락을 구합니다.

### 참여 조건

- 머신러닝과 딥러닝의 기본 원리를 이해하고 PyTorch 학습 코드를 읽고 수정할 수 있어야 합니다.
- Python 개발 및 Git/GitHub 사용 경험이 필요합니다.
- 강화학습, 로보틱스, MuJoCo, VLM/VLA 경험은 필수가 아니지만 강의와 실습 난이도가 높습니다. 새로운 기술을 스스로 조사하고 문제를 해결할 수 있어야 합니다.
- Physical AI, 로보틱스, VLM/VLA 분야로 연구 또는 커리어를 확장하려는 분을 환영합니다.

### AI 사용 규정

학습, 자료 조사, 번역, 코드 작성과 디버깅 과정에서 AI 및 AI agent 사용을 권장합니다.

단, AI가 생성한 내용을 이해하지 않은 채 발표에서 읽거나 코드와 문서를 그대로 복사하는 행위는 허용하지 않습니다. 참여자는 자신이 제출하거나 발표한 결과물의 동작 원리, 선택 이유, 한계와 오류 가능성을 직접 설명할 수 있어야 합니다.

필요하면 질의응답, 코드 수정 또는 재현을 통해 이해 여부를 확인합니다. 규정 위반 시 1회 경고하며, 2회 확인되면 스터디에서 하차합니다. AI 사용 여부가 아니라 AI 결과를 검증하고 자신의 지식으로 전환했는지를 평가합니다.

## 📚 Archive

학습 자료, 발표, 과제 수행 기록, 오픈소스 재현 및 기여 내역을 활동에 따라 추가합니다. 아직 공개된 결과물은 없습니다.

## 🌱 How to Engage

참여 안내는 [프로젝트 페이지](https://pseudo-lab.com/projects/326b9bb8-c161-4757-ace1-09da00c852d9)와 [가짜연구소 Discord](https://discord.gg/EPurkHVtp2)를 확인해주세요. 정기 모임 장소는 Room-CS입니다.

## Acknowledgement 🙏

이 프로젝트는 가짜연구소 Open Academy로 진행됩니다. 참여자와 오픈소스 기여자들의 지식 공유와 협업에 감사드립니다.

## About Pseudo Lab 👋🏼

[가짜연구소](https://pseudo-lab.com/)는 Sharing, Motivation, Collaborative Joy를 바탕으로 함께 성장하는 비영리 AI 커뮤니티입니다.

## Contributors 😃

[기여자 목록](https://github.com/Pseudo-Lab/robot-learning-for-physical-ai/graphs/contributors)

## License 🗞

이 저장소에서 프로젝트 구성원이 직접 작성한 코드와 문서는 [MIT License](LICENSE)를 따릅니다. 외부 강의 자료, 과제, 논문, 모델, 데이터셋 및 오픈소스 코드는 각 원저작물의 라이선스와 이용 조건을 따릅니다.
