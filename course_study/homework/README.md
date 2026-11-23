# ETH Robot Learning · HW1–HW4

가짜연구소 스터디에서 ETH Zurich의 2026년 봄 Robot Learning 과제 14문제를 구현하고 평가했습니다.

| 과제 | 원본 | 구현과 평가 |
|---|---|---|
| HW1 · PyTorch | [과제 원문](original/hw1/) | [텐서 연산](implementations/hw1/#ex1) · [학습 도구](implementations/hw1/#ex2) · [MNIST 분류](implementations/hw1/#ex3) · [TinyViT](implementations/hw1/#ex4) |
| HW2 · 로봇 제어 | [과제 원문](original/hw2/) | [역기구학](implementations/hw2/#ex1) · [PID](implementations/hw2/#ex2) · [PPO 위치 제어](implementations/hw2/#ex3) |
| HW3 · Imitation learning | [과제 원문](original/hw3/) | [단일 큐브](implementations/hw3/#ex1) · [DAgger와 장애물](implementations/hw3/#ex2) · [다중 큐브](implementations/hw3/#ex3) |
| HW4 · 강화학습 | [과제 원문](original/hw4/) | [GridWorld](implementations/hw4/#ex1) · [DQN](implementations/hw4/#ex2) · [PPO](implementations/hw4/#ex3) · [SAC](implementations/hw4/#ex4) |

## 실행용 사본 만들기

Linux에서 Python 3.12와 Bash를 사용합니다. 저장소 최상위 폴더에서 실행하세요. `PUB`는 코드와 결과를 담은 폴더이고 `R`은 실행용 사본을 만들 경로입니다. 공식 저장소를 이미 내려받았다면 기준 커밋이 있는 해당 경로를 `prepare.py`의 첫 번째 인수로 지정하세요.

```bash
export PUB="$PWD/course_study/homework"
export R="$PWD/homework-run"
git clone https://github.com/mees-robot-learning-course/ethz-course-2026.git ethz-course-2026
python3.12 "$PUB/prepare.py" ethz-course-2026 "$R"
```

`prepare.py`는 공식 과제 코드에 `changes/`의 변경분(`.rcs`)을 적용하고, `files/`의 Python 모듈·테스트·평가용 정책을 복사합니다. `R`에는 새 폴더 이름을 지정하세요. 실행용 사본은 원래 패키지 이름과 경로를 유지하며, HW1 노트북 네 개는 Jupyter에서 열 수 있습니다. [실행용 사본 생성 스크립트](prepare.py) · [공식 기준 커밋](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/2e089e97d84968182c1ffa2e1bc8177530feb6fe)

## 환경과 검사

실험에는 Ubuntu 24.04 WSL, Python 3.12, PyTorch 2.11.0, torchvision 0.26.0을 사용했습니다. GPU 학습에는 CUDA 12.8용 PyTorch를 설치했습니다. 기능 테스트와 저장된 정책의 재평가는 CPU에서 실행할 수 있습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$PUB/requirements.txt"
cd "$R"
MUJOCO_GL=egl HW_DEVICE=cpu python -m pytest -q tests
```

CPU에서 테스트 30개가 통과했습니다. 정책을 다시 평가하는 명령은 HW3와 HW4 문서에 있습니다. [HW3 채점](implementations/hw3/) · [HW4 로봇팔 평가](implementations/hw4/)
