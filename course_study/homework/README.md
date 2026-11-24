# ETH Robot Learning · HW1–HW4

가짜연구소 스터디에서 ETH Zurich의 2026년 봄 Robot Learning 과제 14문제를 구현하고 평가했습니다.

| 풀이와 시연 | 다루는 내용 |
|---|---|
| [HW1 · PyTorch](implementations/hw1/) | 텐서 연산과 자동미분, MNIST 분류, TinyViT |
| [HW2 · 로봇 제어](implementations/hw2/) | 역기구학, PID 경로 추종, PPO를 이용한 목표 위치 제어 |
| [HW3 · Imitation learning](implementations/hw3/) | 단일 큐브 놓기, 장애물 앞에서의 DAgger, 색으로 지정한 큐브 놓기 |
| [HW4 · 강화학습](implementations/hw4/) | GridWorld, CartPole의 DQN, 로봇팔의 PPO와 SAC |

## 풀이 실행하기

`implementations/hwN/src/`에는 풀이·실험 코드와 테스트가 있고, `results/`에는 평가 결과가 있습니다. 시뮬레이터, 로봇 모델, 기본 학습 클래스와 HW3 채점기는 ETH 과제에서 가져옵니다.

[ETH Robot Learning 공식 저장소](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/2e089e97d84968182c1ffa2e1bc8177530feb6fe)

Linux에서 저장소 최상위 폴더를 기준으로 실행하세요. `R`은 새로 만들 실행 폴더입니다. 아래 명령은 ETH 원본을 `$R/ethz_starter/`에 내려받고, `$R`에 복사한 과제에 풀이 파일을 덮어씁니다.

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

CPU에서 테스트 30개가 통과했습니다.

<details>
<summary>ETH 강의 영상</summary>

- HW2 · [L02 · Robot Control & MDPs](https://www.youtube.com/watch?v=5-Bb84eTTqQ)
- HW3 · [L03 · Imitation Learning](https://www.youtube.com/watch?v=Ef4R5s1LqoQ) · [L07 · Action chunking](https://www.youtube.com/watch?v=imSTfMJjp7M)
- HW4 · [L04 · Reinforcement Learning I](https://www.youtube.com/watch?v=90raNpc11tQ) · [L05 · Reinforcement Learning II](https://www.youtube.com/watch?v=AdTGz8YnnlE)

</details>
