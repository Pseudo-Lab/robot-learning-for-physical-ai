# HW4 · GridWorld에서 SO100 로봇팔까지

GridWorld에서는 상태 전이 확률로 행동을 고르고, CartPole에서는 경험으로 이산 행동의 가치를 학습합니다. 로봇팔 과제에서는 PPO와 SAC가 여섯 관절에 보낼 연속 제어 명령을 학습합니다.

[L04 · Reinforcement Learning I](https://www.youtube.com/watch?v=90raNpc11tQ) · [L05 · Reinforcement Learning II](https://www.youtube.com/watch?v=AdTGz8YnnlE) · [공식 원본](../../original/hw4/) · [실행 환경 준비](../../)

<a id="ex1"></a>

## Ex1 · GridWorld

같은 칸에서 같은 행동을 해도 미끄러지면 다른 칸에 도착합니다. L04의 Bellman 계산은 가능한 전이를 모두 고려해 행동 가치를 구합니다.

`action_values`는 `env.P[state][action]`에서 전이 확률, 다음 칸, reward, 종료 여부를 읽습니다. 각 경우의 `reward + gamma * values[nxt]`를 확률로 가중해 더하고, 종료되는 전이에서는 즉시 reward만 남깁니다.

[전이 확률로 행동 가치 계산](src/hw4_reinforcement_learning/exercises/ex1_mdp.py#L4-L12)

두 알고리즘은 이 행동 가치들을 다르게 사용합니다. **Policy iteration**은 현재 policy의 행동 확률로 평균을 내어 상태 가치를 평가한 뒤, 가치가 가장 큰 행동들에 같은 확률을 배분합니다.

**Value iteration**은 매 갱신에서 곧바로 행동 가치의 최댓값을 현재 칸의 새 가치로 씁니다.

[policy 평가와 개선](src/hw4_reinforcement_learning/exercises/ex1_mdp.py#L22-L47) · [최댓값을 쓰는 갱신](src/hw4_reinforcement_learning/exercises/ex1_mdp.py#L54-L65)

두 구현 모두 입력 100개에서 정확성 검사를 통과했습니다. 별도 실험에서는 100스텝 안에 목표에 닿을 확률을 최대화하는 policy를 구했습니다. 미끄러짐 확률이 0.2일 때 그 이론적 상한은 **90.0017%**이며, 고정 시드 100회에서 91회 도달했습니다.

[구현](src/hw4_reinforcement_learning/exercises/ex1_mdp.py) · [유한 시간 도달 확률 계산](src/experiments/hw4/gridworld.py) · [평가 결과](results/grid_optimal.json)

<details>
<summary>GridWorld에서 한 경로 따라가기</summary>

미끄러질 확률이 0.2인 격자에서 100스텝 안에 목표에 닿을 확률을 최대화하는 policy를 실행했습니다. 점은 시드 107000에서 실제로 방문한 칸을 따라갑니다.

![미끄러짐이 있는 격자에서 시작점과 목표 사이를 이동하는 경로](media/gridworld.gif)

</details>

<a id="ex2"></a>

## Ex2 · CartPole에서 DQN

CartPole에서는 관측을 `QNet`에 넣으면 카트를 왼쪽 또는 오른쪽으로 밀었을 때의 기대 누적 reward가 각각 나옵니다. L04의 DQN(Deep Q-Network)은 replay buffer에 저장한 경험 $(s,a,r,s',d)$로 Q값을 학습합니다.

목표값 $y$는 즉시 받은 reward $r$과 할인율 $\gamma$를 적용한 다음 상태의 최대 Q값을 더한 값입니다. 저장된 종료 표시 $d$가 1이면 $y=r$입니다.

[경험 저장과 Q값 출력](src/hw4_reinforcement_learning/exercises/ex2_dqn.py#L7-L14)

$$
y=r+\gamma(1-d)\max_{a'}Q_{\mathrm{target}}(s',a')
$$

`DQN.update`에서 `target_q_net(next_states).amax(dim=1)`은 다음 상태에서의 최대 Q값입니다. 현재 네트워크의 출력에서는 `gather`로 **실제로 실행한 행동**의 Q값만 꺼내 목표값 $y$와의 평균 제곱 오차를 줄입니다. 현재 네트워크를 갱신해도 목표값을 계산하는 네트워크는 일정 기간 고정하고, 정해진 간격에만 가중치를 복사합니다.

[TD 목표와 가중치 갱신](src/hw4_reinforcement_learning/exercises/ex2_dqn.py#L41-L60)

저장된 policy는 100개 에피소드 중 **99개**에서 누적 reward 475 이상을 기록했습니다.

[DQN 구현](src/hw4_reinforcement_learning/exercises/ex2_dqn.py) · [시드별 reward](results/cartpole.json)

<details>
<summary>DQN 추가 학습 전·중·후의 균형 잡기</summary>

기존 DQN에서 시작한 추가 학습을 같은 시드로 재현했습니다. 같은 초기 상태(시드 107100)에서 균형을 유지한 시간은 추가 학습 전 286스텝, 36 episode 후 452스텝, 72 episode 후 500스텝입니다. 각 패널은 그 시행이 끝난 자세에서 멈춥니다.

![같은 시작 상태에서 DQN 추가 학습 횟수에 따라 막대가 넘어지는 시점 비교](media/dqn.gif)

</details>

<a id="ex3"></a>

## Ex3 · PPO로 로봇팔 제어

PPO는 현재 policy로 모은 행동이 그 상태에서 기대한 것보다 얼마나 나았는지 나타내는 *advantage*를 사용합니다. `select_action`은 뽑은 행동의 로그 확률밀도와 critic의 상태 가치를 함께 반환합니다.

학습 루프는 이 값을 rollout buffer에 저장하고, 수집이 끝나면 `compute_returns`로 reward와 상태 가치에서 advantage를 계산합니다.

[행동 샘플링](src/hw4_reinforcement_learning/exercises/ex3_ppo.py#L11-L19) · [rollout 수집](src/hw4_reinforcement_learning/scripts/train_ppo.py#L54-L79) · [ETH 과제의 advantage 계산](https://github.com/mees-robot-learning-course/ethz-course-2026/blob/2e089e97d84968182c1ffa2e1bc8177530feb6fe/hw4_reinforcement_learning/rl/buffers.py#L128-L144)

L05의 확률밀도 비율 $\rho=\pi_{\rm new}(a\mid s)/\pi_{\rm old}(a\mid s)$가 코드에서는 `exp(logp_batch - old_logp_batch)`입니다. `compute_surrogate_loss`는 $\rho A$와 $\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)A$ 중 작은 값을 고릅니다. 그 평균에 음수를 붙이고 `surrogate_loss_coeff`를 곱하면 policy 손실입니다.

예를 들어 $A=2$, $\epsilon=0.2$, $\rho=1.4$라면 두 항은 2.8과 2.4입니다. 이 표본에서 좋은 행동의 확률을 더 높여도 목적함수에 반영되는 이득은 2.4에서 멈춥니다.

[PPO clipping과 손실](src/hw4_reinforcement_learning/exercises/ex3_ppo.py#L31-L49)

시드 42로 1,024,000 환경 스텝 동안 학습했습니다.

[PPO 구현](src/hw4_reinforcement_learning/exercises/ex3_ppo.py) · [순수 PPO의 조건과 장면별 오차](results/rl-ppo/)

<a id="demo"></a>

### PPO 학습 단계별 로봇팔 움직임

고정 평가의 첫 장면(시드 107400)을 사용했습니다. 세 패널은 같은 시작 자세에서 노란 원으로 표시한 목표를 3초간 추종합니다. 왼쪽부터 0, 409,600, 1,024,000 환경 스텝에서 저장한 모델입니다. 아래 숫자는 해당 순간의 손끝 오차입니다.

![순수 PPO의 학습 전·중·후 로봇팔 동작](media/rl-ppo.gif)

<details>
<summary>같은 장면에서 BC + PPO 보기</summary>

시연으로 행동을 먼저 학습한 뒤, 같은 세 시점에서 PPO를 비교했습니다. 왼쪽 패널은 BC를 마치고 PPO를 시작하는 시점입니다.

![BC로 초기화한 PPO의 학습 전·중·후 동작](media/bc-ppo.gif)

</details>

<a id="ex4"></a>

## Ex4 · SAC로 같은 로봇팔 제어

SAC(Soft Actor-Critic)는 replay buffer에 저장한 전이를 재사용합니다. L05의 maximum-entropy 목표는 기대 누적 reward와 행동 분포의 entropy를 함께 높이는 것입니다. `compute_critic_loss`는 다음 상태 $s'$에서 현재 actor로 행동 $a'$를 새로 뽑고 두 target critic의 추정치 중 작은 값으로 다음 목표를 계산합니다.

$$
y=r+\gamma(1-d)\left[\min_{i=1,2}Q_{\mathrm{target},i}(s',a')-\alpha\log\pi(a'\mid s')\right]
$$

두 critic은 각각 저장된 행동의 Q값을 $y$에 맞춥니다. `train_sac.py`는 `done=terminal`을 buffer에 저장합니다.

시간 제한으로 장면을 초기화한 경우에는 $d=0$이므로 목표값에 다음 상태의 가치가 남습니다.

[critic의 목표와 손실](src/hw4_reinforcement_learning/exercises/ex4_sac.py#L16-L26) · [종료 표시 저장과 replay 샘플링](src/hw4_reinforcement_learning/scripts/train_sac.py#L63-L77)

`compute_actor_loss`는 현재 상태에서 actor가 새로 뽑은 행동에 대해 `alpha * logp_new - q_new`의 평균을 최소화합니다. $\alpha$는 탐색에 주는 가중치입니다.

`compute_alpha_loss`로 목표 entropy에 맞춰 이 값을 조정하고, `soft_update_targets`로 현재 critic의 가중치를 target critic에 조금씩 반영합니다.

[actor·온도·target 갱신](src/hw4_reinforcement_learning/exercises/ex4_sac.py#L28-L65)

PPO와 같은 시드 42로 1,024,500 환경 스텝 동안 학습했습니다. PPO와 SAC의 총 환경 스텝 수는 과제 기본값과 같습니다. SAC는 128스텝마다 크기 256의 배치로 48회 업데이트했습니다.

[SAC 구현](src/hw4_reinforcement_learning/exercises/ex4_sac.py) · [GPU 학습 실행기](src/experiments/hw4/train_gpu.py) · [순수 SAC의 조건과 장면별 오차](results/rl-sac/)

### SAC 학습 단계별 로봇팔 움직임

PPO와 같은 장면에서 0, 409,600, 1,024,500 환경 스텝의 SAC 모델을 재생했습니다.

![순수 SAC가 PPO와 같은 목표를 추종하는 세 단계의 동작](media/rl-sac.gif)

<details>
<summary>같은 장면에서 BC + SAC 보기</summary>

왼쪽은 BC를 마친 모델이고, 가운데와 오른쪽은 SAC로 추가 학습한 모델입니다.

![BC로 초기화한 SAC의 학습 전·중·후 동작](media/bc-sac.gif)

</details>

## 로봇팔의 성공 기준

3초 동안 0.1초마다 측정한 **말단 위치 오차 30개의 평균이 5cm 미만**이면 해당 장면을 성공으로 집계했습니다. 고정 평가는 시드 107400~107499를 사용합니다. 보충 평가에서는 팔을 최대로 뻗어도 목표의 5cm 안에 닿지 않는 세 장면(107405·107410·107423)을 다음 세 시드(107500~107502)의 장면으로 교체했습니다.

순수 PPO와 순수 SAC는 무작위 가중치에서 학습을 시작했습니다. BC + PPO와 BC + SAC는 역기구학 시연 2,048개로 평균 행동을 먼저 학습했습니다. 네 policy 모두 GPU에서 학습한 뒤 CPU에서 평가했습니다.

| 최종 policy | 고정 100개 | 보충 100개 | 시드 107492의 3초 평균 오차 |
|---|---:|---:|---:|
| 순수 PPO | 95회 | **98회** | 5.765cm |
| BC + PPO | 93회 | **96회** | 6.298cm |
| 순수 SAC | 92회 | **95회** | 9.037cm |
| BC + SAC | 92회 | **95회** | 9.659cm |

이전 다단계 혼합 학습에서는 PPO와 SAC 모두 보충 100개 장면 중 98개에서 성공했습니다.

[이전 실험 결과](results/mixed_reference.json) · [학습 스텝과 checkpoint별 결과](results/robot_arm.json)

<a id="저장된-정책-재평가"></a>

## 저장된 policy 재평가

```bash
cd "$R"
python -m pytest -q tests/test_hw4.py
python experiments/hw4/cartpole.py
python experiments/hw4/evaluate.py "$PUB/implementations/hw4/results"
```

CartPole은 `99/100`을 출력합니다. 로봇팔 평가는 policy마다 103개 장면을 재현합니다. 각 장면의 초기 관측과 목표가 저장값과 같고, 평균 오차와 마지막 시점의 오차가 각각 저장값과 0.000001m 이내로 일치하면 통과합니다.

## GPU 학습 조건 재사용

`train_gpu.py`는 PyTorch 2.11.0+cu128에서 실행합니다. 새 실험 폴더에서 `python "$R/experiments/hw4/train_gpu.py" ppo rl`처럼 알고리즘(`ppo`, `sac`)과 초기화(`rl`, `bc`)를 선택합니다. BC 조건은 같은 폴더에서 먼저 `python "$R/experiments/hw4/generate_demos.py"`를 실행해 시연을 생성합니다.
