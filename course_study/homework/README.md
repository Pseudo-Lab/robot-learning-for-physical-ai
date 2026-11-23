# ETH Robot Learning · HW1–HW4

가짜연구소 스터디에서 ETH Zurich의 2026년 봄 Robot Learning 과제 14문제를 구현하고 평가했습니다.

| 과제 | 원본 | 구현과 평가 |
|---|---|---|
| HW1 · PyTorch | [과제 원문](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/2e089e97d84968182c1ffa2e1bc8177530feb6fe/hw1_pytorch_tutorial) | [텐서 연산](implementations/hw1/#ex1) · [학습 도구](implementations/hw1/#ex2) · [MNIST 분류](implementations/hw1/#ex3) · [TinyViT](implementations/hw1/#ex4) |
| HW2 · 로봇 제어 | [과제 원문](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/2e089e97d84968182c1ffa2e1bc8177530feb6fe/hw2_robot_control_mdps) | [역기구학](implementations/hw2/#ex1) · [PID](implementations/hw2/#ex2) · [PPO 위치 제어](implementations/hw2/#ex3) |
| HW3 · Imitation learning | [과제 원문](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/2e089e97d84968182c1ffa2e1bc8177530feb6fe/hw3_imitation_learning) | [단일 큐브](implementations/hw3/#ex1) · [DAgger와 장애물](implementations/hw3/#ex2) · [다중 큐브](implementations/hw3/#ex3) |
| HW4 · 강화학습 | [과제 원문](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/2e089e97d84968182c1ffa2e1bc8177530feb6fe/hw4_reinforcement_learning) | [GridWorld](implementations/hw4/#ex1) · [DQN](implementations/hw4/#ex2) · [PPO](implementations/hw4/#ex3) · [SAC](implementations/hw4/#ex4) |

## 풀이 실행하기

각 과제의 ‘과제 원문’ 링크는 ETH 공식 저장소로, 각 문제 링크는 풀이 설명과 소스 파일로 연결됩니다. `implementations/hwN/src/`에는 풀이·실험 코드와 테스트가 있고, `results/`에는 평가 결과가 있습니다. 시뮬레이터, 로봇 모델, 기본 학습 클래스와 HW3 채점기는 [ETH Robot Learning 과제 저장소](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/2e089e97d84968182c1ffa2e1bc8177530feb6fe)를 사용합니다.

Linux에서 이 저장소의 최상위 폴더를 열고 다음 명령을 실행하세요. `R`에는 새 디렉터리를 지정합니다. 공식 코드는 `ethz_starter/`에 두고, 실행할 사본에 과제별 풀이 파일을 복사합니다.

```bash
export PUB="$PWD/course_study/homework"
export R="$PWD/homework-run"
mkdir "$R" && (
    set -e
    git clone https://github.com/mees-robot-learning-course/ethz-course-2026.git "$R/ethz_starter"
    git -C "$R/ethz_starter" checkout --detach 2e089e97d84968182c1ffa2e1bc8177530feb6fe
    cp -R "$R/ethz_starter"/hw*/ "$R/"
    for hw in "$PUB"/implementations/hw*; do cp -R "$hw/src/." "$R/"; done
    cp "$PUB/conftest.py" "$R/tests/"
)
export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"
```

HW1 노트북 네 개는 Jupyter에서 바로 열 수 있습니다. HW2~HW4는 내려받은 공식 과제의 환경과 자산을 사용합니다.

## 환경과 검사

실험에는 Ubuntu 24.04 WSL, Python 3.12, PyTorch 2.11.0, torchvision 0.26.0을 사용했습니다. GPU 학습에는 CUDA 12.8용 PyTorch를 설치했습니다. 기능 테스트와 저장된 policy의 평가는 CPU에서 실행할 수 있습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$PUB/requirements.txt"
cd "$R"
MUJOCO_GL=egl HW_DEVICE=cpu python -m pytest -q tests
```

CPU에서 테스트 30개가 통과했습니다. [HW3 채점](implementations/hw3/) · [HW4 로봇팔 평가](implementations/hw4/)
