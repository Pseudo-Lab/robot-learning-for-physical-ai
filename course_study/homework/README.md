# ETH Robot Learning · HW1–HW4

ETH Zurich의 2026년 봄 Robot Learning 과제를 따라 가짜연구소 스터디에서 구현하고 평가한 14개 문제입니다. 원본 링크는 실험에 사용한 공식 커밋에 고정했습니다. 각 구현 문서에서는 문제를 푸는 원리, 해당 코드, 실행 명령과 측정 결과를 함께 볼 수 있습니다.

| 과제 | 원본 | 구현과 평가 |
|---|---|---|
| HW1 · PyTorch | [과제 원문](original/hw1/) | [텐서 연산](implementations/hw1/#ex1) · [학습 도구](implementations/hw1/#ex2) · [MNIST 분류](implementations/hw1/#ex3) · [TinyViT](implementations/hw1/#ex4) |
| HW2 · 로봇 제어 | [과제 원문](original/hw2/) | [역기구학](implementations/hw2/#ex1) · [PID](implementations/hw2/#ex2) · [PPO 위치 제어](implementations/hw2/#ex3) |
| HW3 · Imitation learning | [과제 원문](original/hw3/) | [단일 큐브](implementations/hw3/#ex1) · [DAgger와 장애물](implementations/hw3/#ex2) · [다중 큐브](implementations/hw3/#ex3) |
| HW4 · 강화학습 | [과제 원문](original/hw4/) | [GridWorld](implementations/hw4/#ex1) · [DQN](implementations/hw4/#ex2) · [PPO](implementations/hw4/#ex3) · [SAC](implementations/hw4/#ex4) |

## 실행용 사본 만들기

Linux의 Bash와 Python 3.12를 기준으로 합니다. **이 스터디 저장소의 최상위 폴더**에서 시작하세요. `PUB`는 공개된 코드·결과 폴더, `R`은 새로 만들 실행용 사본입니다. 이미 받은 공식 저장소에 기준 커밋이 있다면 그 경로를 `prepare.py`의 첫 번째 인수로 사용할 수 있습니다.

```bash
export PUB="$PWD/course_study/homework"
export R="$PWD/homework-run"
git clone https://github.com/mees-robot-learning-course/ethz-course-2026.git ethz-course-2026
python3.12 "$PUB/prepare.py" ethz-course-2026 "$R"
```

`prepare.py`는 Git에서 기준 커밋의 파일을 읽고 새 폴더에 구현을 조립합니다. `changes/`에는 스터디에서 바꾼 코드 줄과 편집 위치를 담은 텍스트 파일(`.rcs`)이 있습니다. `files/`에는 추가한 Python 모듈, 테스트와 평가용 정책이 있습니다. 조립하면 원래 패키지 이름과 실행 경로를 그대로 쓸 수 있고, HW1의 네 노트북은 Jupyter에서 열 수 있습니다. 이미 존재하는 출력 경로를 지정하면 중단합니다. [조립 코드](prepare.py) · [공식 기준 커밋](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/2e089e97d84968182c1ffa2e1bc8177530feb6fe)

## 환경과 검사

기존 실험 환경은 Ubuntu 24.04 WSL, Python 3.12, PyTorch 2.11.0, torchvision 0.26.0입니다. 새 가상환경에서는 아래와 같이 의존성을 설치합니다. GPU 학습에는 CUDA 12.8용 PyTorch 빌드를 사용했습니다. CPU 기능 검사와 저장된 정책의 재평가에는 GPU가 필요하지 않습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$PUB/requirements.txt"
cd "$R"
MUJOCO_GL=egl HW_DEVICE=cpu python -m pytest -q tests
```

공개용 사본에서 과제 테스트 30개가 통과했습니다. HW3 과제 채점과 HW4 로봇팔 성능은 각 과제 문서에 있는 명령으로 저장된 정책을 불러와 확인할 수 있습니다. MNIST 정확도와 로봇팔의 GPU 학습 결과는 기존 실행 기록을 사용했습니다. 실제 로봇 구동과 수강생의 외부 제출은 평가 범위에서 제외했습니다.

실험 당시 로컬 커밋은 `f1cdbac`이고 작업 트리에 미커밋 변경이 있었습니다. 공개본에는 현재 노트북·구현의 변경분과 결과를 산출한 별도 정책을 함께 실었습니다. HW4에서 **96/100은 스터디가 정한 목표**입니다. 도달 가능한 100개 장면에서 순수 PPO는 98회, 순수 SAC는 95회 성공했습니다. [HW4 평가 조건과 비교 결과](implementations/hw4/)
