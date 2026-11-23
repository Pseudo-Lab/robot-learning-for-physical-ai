# HW2 · SO100 로봇팔의 위치 제어

SO100 로봇팔이 주어진 경로를 따라가도록 역기구학과 PID 제어기를 구현했습니다. PPO 실험에서는 관절 상태와 목표 위치를 보고 행동을 고르는 policy를 학습했습니다. [L02 · Robot Control & MDPs](https://www.youtube.com/watch?v=5-Bb84eTTqQ) · [공식 원본](../../original/hw2/) · [실행 환경 준비](../../)

<a id="ex1"></a>
## Ex1 · 역기구학

8자 경로의 목표점은 손끝의 좌표로 주어지지만, 로봇에 필요한 값은 관절 자세 $q$입니다. L02의 선형 근사 $\Delta x \approx J(q)\Delta q$는 지금 자세에서 관절을 조금 움직였을 때 손끝이 어디로 갈지 예측합니다. `build_keypoints`는 $x$를 고정하고 $y,z$ 평면에 목표점을 배치합니다. `ik_track`은 목표점에서 현재 손끝 위치를 뺀 `error`와 MuJoCo가 계산한 위치 Jacobian $J$로 다음 보정 방향을 구합니다. [경로 생성과 IK 반복문](src/hw2_robot_control_mdps/exercises/ex1.py#L5-L45)

$$
(JJ^\top + \lambda I)u = k\,\mathrm{error}, \qquad \mathrm{delta} = J^\top u
$$

Damped least squares 식에서 $\lambda$는 `damping`, $I$는 세 위치 축에 대한 단위행렬, $k$는 `pos_gain`입니다. 감쇠항은 특이 자세 근처에서 관절 보정량이 지나치게 커지는 것을 줄입니다. `np.linalg.solve`로 $u$를 구한 뒤 `delta`의 각 성분을 제한하고 `dt`를 곱해 관절값을 갱신합니다. 관절 한계를 적용한 새 자세에서 손끝 오차를 다시 계산합니다.

폭 0.14~0.25m인 궤적 100개를 시험했습니다. 모든 궤적에서 관절 한계를 지켰고, 궤적마다 목표점 16곳의 위치 오차가 모두 5mm 미만이었습니다. [역기구학 구현](src/hw2_robot_control_mdps/exercises/ex1.py) · [궤적별 최대 오차](results/ex1.json)

<a id="ex2"></a>
## Ex2 · PID 추종

두 점 사이를 직선으로 잇더라도, 구간 안에서 얼마나 빨리 이동할지는 따로 정해야 합니다. `generate_quintic_spline_waypoints`는 구간의 진행률 $s$를 0에서 1까지 나누고 $h(s)=10s^3-15s^4+6s^5$로 바꿉니다. 목표 위치는 `start + h * (end - start)`입니다. $h$의 1차·2차 미분은 양 끝에서 0이므로, 진행률 $s$를 시간에 비례해 늘리면 시작과 끝의 목표 속도와 가속도가 모두 0입니다. [5차 보간](src/hw2_robot_control_mdps/exercises/ex2.py#L4-L9)

`track`은 보간한 각 점을 Ex1의 IK로 풀어 목표 관절값을 미리 구합니다. 제어 중에는 목표 관절값에서 측정값 `data.qpos`를 뺀 오차를 최근 10개까지 보관합니다. `pid_control`은 현재 오차(P), 보관한 오차의 합에 시간 간격을 곱한 값(I), 직전 두 오차의 차이를 시간 간격으로 나눈 값(D)을 조합해 토크 명령을 만듭니다. 기본 설정은 `Ki=0.0`이므로 실제 명령에는 P항과 D항이 쓰입니다. [관절 목표와 피드백 루프](src/hw2_robot_control_mdps/scripts/pid_control.py#L29-L43) · [PID 계산](src/hw2_robot_control_mdps/exercises/ex2.py#L12-L25)

Ex1과 같은 폭 범위로 궤적 100개를 만들고 궤적마다 경유점 80개를 두었습니다. 로봇팔은 모든 경유점에 3cm 이내로 접근했습니다. [PID 구현](src/hw2_robot_control_mdps/exercises/ex2.py) · [궤적별 도달 개수](results/ex2.json)

<a id="ex3"></a>
## Ex3 · PPO로 목표 위치 제어

L02의 Markov decision process(MDP)에서 policy의 행동은 다음 상태와 reward에 영향을 줍니다. 이 과제에서는 MuJoCo가 로봇의 상태를 갱신하고, policy는 그중 아래 관측 벡터를 입력으로 받습니다.

1. **관측.** `get_obs`는 관절값에 베이스 좌표계에서 본 손끝 위치·자세와 목표 위치를 붙입니다. 예를 들어 손끝의 월드 좌표에서 베이스 위치를 빼고 베이스 회전행렬의 전치를 곱하면, 로봇을 기준으로 한 손끝 위치가 됩니다. [관측 벡터 구성](src/hw2_robot_control_mdps/exercises/ex3.py#L39-L55)
2. **행동과 전이.** Stable-Baselines3의 PPO(Proximal Policy Optimization)가 출력한 행동을 `process_action`이 관절별 목표값으로 바꿉니다. 행동 성분이 $-1, 0, 1$이면 각각 그 관절의 하한, 중간값, 상한에 대응합니다. `SO100TrackEnv.step`은 이 목표값을 MuJoCo에 전달해 팔을 움직인 뒤 다음 관측을 읽습니다. [행동 변환](src/hw2_robot_control_mdps/exercises/ex3.py#L25-L30) · [환경 한 스텝](src/hw2_robot_control_mdps/env/so100_tracking_env.py#L72-L85) · [PPO 학습](src/hw2_robot_control_mdps/scripts/train.py#L60-L85)
3. **Reward.** 손끝과 목표 사이의 거리가 $d$미터일 때 `compute_reward`는 $e^{-2d}$를 주고, $d<0.005$이면 1을 더합니다. 가까워질수록 받는 보상과 5mm 안에 들어왔을 때의 추가 보상을 합친 값이 매 스텝의 학습 신호입니다. [Reward 계산](src/hw2_robot_control_mdps/exercises/ex3.py#L33-L36)

평가에서는 시드 107200~107299의 각 장면에서 policy가 예측한 평균 행동을 실행했습니다. **2초가 끝났을 때** 팔 끝과 목표 사이의 거리가 5cm 미만인 장면은 **100개 중 99개**였습니다. [관측과 reward 구현](src/hw2_robot_control_mdps/exercises/ex3.py) · [평가 코드와 policy](src/experiments/hw2/) · [장면별 최종 오차](results/ex3.json)

### 실행

```bash
cd "$R"
python -m pytest -q tests/test_hw2.py tests/test_hw2_integration.py
mkdir hw2-check && cd hw2-check
for n in 1 2 3; do
    python "$R/experiments/hw2/evaluate.py" "$n" --max-width .25
done
```

출력의 세 숫자는 문제 번호, 성공 횟수, 총 시도 횟수입니다. 각 문제의 결과는 `1 100 100`, `2 100 100`, `3 99 100`입니다.
