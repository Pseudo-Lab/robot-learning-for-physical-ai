# HW2 · SO100 로봇팔의 위치 제어

SO100 로봇팔이 주어진 경로를 따라가도록 역기구학과 PID 제어기를 구현했습니다. PPO 실험에서는 관절 상태와 목표 위치를 보고 행동을 고르는 policy를 학습했습니다. [공식 원본](../../original/hw2/) · [실행 환경 준비](../../)

<a id="ex1"></a>
## Ex1 · 역기구학

Jacobian은 관절 각도가 변할 때 로봇팔 끝이 어느 방향으로 얼마나 움직이는지 나타냅니다. 이를 이용해 목표점까지의 오차를 줄이는 관절 변화량을 구합니다. 특이 자세 부근에서 작은 위치 오차를 보정할 때 관절이 지나치게 움직이지 않도록 damped least squares를 사용했습니다. 폭 0.14~0.25m인 궤적 100개를 시험했습니다. 모든 궤적에서 관절 한계를 지켰고, 궤적마다 목표점 16곳의 위치 오차가 모두 5mm 미만이었습니다. [역기구학 구현](src/hw2_robot_control_mdps/exercises/ex1.py) · [궤적별 최대 오차](results/ex1.json)

<a id="ex2"></a>
## Ex2 · PID 추종

경유점(waypoint) 사이를 5차 다항식으로 연결하고 역기구학으로 각 점의 관절 목표를 구합니다. PID 제어기는 현재 관절 오차, 누적 오차와 오차 변화량을 토크 명령에 반영합니다. Ex1과 같은 폭 범위로 궤적 100개를 만들고 궤적마다 경유점 80개를 두었습니다. 로봇팔은 모든 경유점에 3cm 이내로 접근했습니다. [PID 구현](src/hw2_robot_control_mdps/exercises/ex2.py) · [궤적별 도달 개수](results/ex2.json)

<a id="ex3"></a>
## Ex3 · PPO로 목표 위치 제어

목표 위치와 팔 끝 사이의 오차로 reward를 계산하고, PPO(Proximal Policy Optimization)로 policy를 학습합니다. 평가에서는 시드 107200~107299의 각 장면에서 policy가 예측한 평균 행동을 실행했습니다. **2초가 끝났을 때** 팔 끝과 목표 사이의 거리가 5cm 미만인 장면은 **100개 중 99개**였습니다. [관측과 reward 구현](src/hw2_robot_control_mdps/exercises/ex3.py) · [평가 코드와 policy](src/experiments/hw2/) · [장면별 최종 오차](results/ex3.json)

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
