# HW4 · GridWorld에서 SO100 로봇팔까지

Ex1에서는 상태 전이 확률을 이용해 각 상태에서 어떤 행동을 고를지 정합니다. Ex2는 경험으로 이산 행동의 가치를 학습하고, Ex3·Ex4는 여섯 관절에 연속 명령을 보내 같은 로봇팔을 움직입니다. [공식 원본](../../original/hw4/) · [실행 환경 준비](../../)

<a id="ex1"></a>
## Ex1 · GridWorld

Policy iteration은 현재 policy로 얻을 누적 reward를 추정한 뒤 각 상태에서 더 유리한 행동을 고릅니다. Value iteration은 각 상태에서 가장 유리한 행동의 가치를 반복해서 갱신합니다. 두 구현 모두 입력 100개에서 정확성 검사를 통과했습니다. 별도 실험에서는 100스텝 안에 목표에 닿을 확률을 최대화하는 policy를 구했습니다. 미끄러짐 확률이 0.2일 때 그 이론적 상한은 **90.0017%**이며, 고정 시드 100회에서 91회 도달했습니다. [구현](src/hw4_reinforcement_learning/exercises/ex1_mdp.py) · [유한 시간 도달 확률 계산](src/experiments/hw4/gridworld.py) · [평가 결과](results/grid_optimal.json)

<a id="ex2"></a>
## Ex2 · CartPole에서 DQN

DQN(Deep Q-Network)은 카트를 왼쪽 또는 오른쪽으로 밀었을 때의 기대 누적 reward를 추정합니다. Replay buffer에서 과거 경험을 뽑고 target network로 학습 목표를 계산합니다. 저장된 policy는 100개 에피소드 중 **99개**에서 누적 reward 475 이상을 기록했습니다. [DQN 구현](src/hw4_reinforcement_learning/exercises/ex2_dqn.py) · [시드별 reward](results/cartpole.json)

<a id="ex3"></a>
## Ex3 · PPO로 로봇팔 제어

PPO는 현재 policy로 경험을 모으고, 각 행동의 결과가 그 상태에서의 기대보다 얼마나 좋았는지 추정합니다. 기대보다 좋았던 행동은 더 자주 선택하도록 학습합니다. 이때 같은 행동에 새 policy와 이전 policy가 부여한 확률의 비율을 구하고, clipping으로 그 비율이 목적함수에 미치는 영향을 제한합니다. 학습 시드 42에서 1,024,000 환경 스텝을 실행했습니다. [PPO 구현](src/hw4_reinforcement_learning/exercises/ex3_ppo.py) · [순수 PPO의 조건과 장면별 오차](results/rl-ppo/)

<a id="ex4"></a>
## Ex4 · SAC로 같은 로봇팔 제어

SAC(Soft Actor-Critic)는 과거 경험을 다시 사용합니다. 두 critic이 추정한 행동 가치 중 작은 값과 policy의 entropy를 이용해, 기대 reward가 큰 행동을 선택하면서도 탐색을 계속하도록 학습합니다. 같은 시드에서 1,024,500 환경 스텝을 실행했습니다. PPO와 SAC의 총 환경 스텝 수는 과제 기본값과 같습니다. SAC는 128스텝마다 크기 256의 배치로 48회 업데이트했습니다. [SAC 구현](src/hw4_reinforcement_learning/exercises/ex4_sac.py) · [GPU 학습 실행기](src/experiments/hw4/train_gpu.py) · [순수 SAC의 조건과 장면별 오차](results/rl-sac/)

### 로봇팔의 성공 기준

3초 동안 0.1초마다 측정한 **말단 위치 오차 30개의 평균이 5cm 미만**이면 해당 장면을 성공으로 집계했습니다. 고정 평가는 시드 107400~107499를 사용합니다. 보충 평가에서는 팔을 최대로 뻗어도 목표의 5cm 안에 닿지 않는 세 장면(107405·107410·107423)을 다음 세 시드(107500~107502)의 장면으로 교체했습니다.

순수 PPO와 순수 SAC는 무작위 가중치에서 학습을 시작했습니다. BC + PPO와 BC + SAC는 역기구학 시연 2,048개로 평균 행동을 먼저 학습했습니다. 네 policy 모두 GPU에서 학습한 뒤 CPU에서 평가했습니다.

| 최종 policy | 고정 100개 | 보충 100개 | 시드 107492의 3초 평균 오차 |
|---|---:|---:|---:|
| 순수 PPO | 95회 | **98회** | 5.765cm |
| BC + PPO | 93회 | **96회** | 6.298cm |
| 순수 SAC | 92회 | **95회** | 9.037cm |
| BC + SAC | 92회 | **95회** | 9.659cm |

이전 다단계 혼합 학습에서는 PPO와 SAC 모두 보충 100개 장면 중 98개에서 성공했습니다. [이전 실험 결과](results/mixed_reference.json) · [학습 스텝과 checkpoint별 결과](results/robot_arm.json)

<a id="저장된-정책-재평가"></a>

### 저장된 policy 재평가

```bash
cd "$R"
python -m pytest -q tests/test_hw4.py
python experiments/hw4/cartpole.py
python experiments/hw4/evaluate.py "$PUB/implementations/hw4/results"
```

CartPole은 `99/100`을 출력합니다. 로봇팔 검사는 policy마다 103개 장면의 초기 관측과 목표를 대조하고, 평균·마지막 오차가 저장값과 0.000001m 이내로 일치해야 통과합니다.

### GPU 학습 조건 재사용

`train_gpu.py`는 PyTorch 2.11.0+cu128에서 실행합니다. 새 실험 폴더에서 `python "$R/experiments/hw4/train_gpu.py" ppo rl`처럼 알고리즘(`ppo`, `sac`)과 초기화(`rl`, `bc`)를 선택합니다. BC 조건은 같은 폴더에서 먼저 `python "$R/experiments/hw4/generate_demos.py"`를 실행해 시연을 생성합니다.
