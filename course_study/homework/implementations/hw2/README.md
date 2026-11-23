# HW2 · SO100 로봇팔의 위치 제어

SO100 로봇팔로 세 가지 제어 방법을 실험했습니다. 역기구학으로 목표점에 맞는 관절 각도를 구하고 PID로 경로를 추종했습니다. PPO 실험에서는 목표 위치에 접근하는 정책을 학습했습니다. [공식 원본](../../original/hw2/) · [실행용 사본 준비](../../)

<a id="ex1"></a>
## Ex1 · 역기구학

말단 위치 오차를 Jacobian으로 관절 변화량에 대응시킵니다. Damped least squares는 특이 자세 부근에서 작은 위치 오차가 큰 관절 변화로 번지는 것을 줄입니다. 폭 0.14~0.25m 범위에서 만든 궤적 100개 모두에서 관절 한계를 지켰고, 각 궤적의 목표점 16개에서 위치 오차가 모두 5mm 미만이었습니다. [역기구학 구현](changes/hw2_robot_control_mdps/exercises/ex1.py.rcs) · [궤적별 최대 오차](results/ex1.json)

<a id="ex2"></a>
## Ex2 · PID 추종

경유점 사이를 5차 다항식으로 연결하고 역기구학으로 각 점의 관절 목표를 구합니다. PID 제어기는 현재 관절 오차, 누적 오차와 오차 변화량을 토크 명령에 반영합니다. 같은 폭 범위의 궤적 100개 모두에서 경유점 80개를 각각 3cm 이내로 추종했습니다. [PID 구현](changes/hw2_robot_control_mdps/exercises/ex2.py.rcs) · [궤적별 도달 개수](results/ex2.json)

<a id="ex3"></a>
## Ex3 · PPO로 목표 위치 제어

로봇의 관절 상태와 목표 위치를 관측하고, 위치 오차에서 보상을 계산해 PPO(Proximal Policy Optimization) 정책을 학습합니다. 평가 때는 시드 107200~107299에서 저장된 정책의 결정적 행동을 실행합니다. **2초 종료 시점**의 말단 오차가 5cm 미만인 장면은 **99/100**입니다. [관측과 보상 구현](changes/hw2_robot_control_mdps/exercises/ex3.py.rcs) · [평가 코드와 정책](files/experiments/hw2/) · [장면별 최종 오차](results/ex3.json)

### 실행

```bash
cd "$R"
python -m pytest -q tests/test_hw2.py tests/test_hw2_integration.py
mkdir hw2-check && cd hw2-check
for n in 1 2 3; do
    python "$R/experiments/hw2/evaluate.py" "$n" --max-width .25
done
```

출력은 차례로 `1 100 100`, `2 100 100`, `3 99 100`입니다.
