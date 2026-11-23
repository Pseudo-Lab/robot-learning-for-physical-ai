# HW3 · 시연으로 큐브 집어 옮기기

로봇이 팔 끝, 그리퍼, 큐브와 목표의 상태를 보고 다음 이동·그리퍼 명령을 정합니다. 기본 구현에는 시연에서 상태와 행동 구간을 뽑는 dataset과 여러 시점의 행동을 한 번에 예측하는 정책이 있습니다. 평가에는 접근·집기·놓기 등 시연 단계별로 다음 행동을 학습한 정책을 사용했습니다. [공식 원본](../../original/hw3/) · [기본 모델 구현](changes/hw3_imitation_learning/hw3/model.py.rcs) · [실행용 사본 준비](../../)

<a id="ex1"></a>
## Ex1 · 단일 큐브 놓기

Behavioral Cloning(BC)은 시연 상태에서 관찰한 행동을 학습합니다. 최종 정책은 접근·집기·들어 올리기·놓기를 이어 가면서 현재 관측으로 다음 단계로 넘어갈 시점을 정합니다. 과제 제공 채점기로 100회 모두 성공해 100점을 받았습니다. 별도 시드에서 큐브가 실제로 놓였는지 검사한 결과는 99/100입니다. [단일 큐브 정책](files/experiments/hw3/single_cube.py) · [독립 평가](results/ex1/)

<a id="ex2"></a>
## Ex2 · DAgger와 장애물

시연에서 조금만 벗어나도 다음 관측이 달라집니다. DAgger(Dataset Aggregation)는 정책이 직접 방문한 상태에서 교정 행동을 추가해 이런 오차 누적을 다룹니다. 시연과 교정 데이터는 스크립트로 생성했습니다. 학습한 정책은 장애물 위치와 현재 조작 단계에 따라 행동을 선택합니다. 과제 채점은 100/100회·100점, 별도 놓기 평가는 **98/100회**입니다. 독립 평가에서 장애물이 왼쪽이면 40/40, 중앙이면 12/14, 오른쪽이면 46/46회 성공했습니다. [시연 수집 코드](files/hw3_imitation_learning/scripts/collect_scripted_demos.py) · [장애물 위치별 결과](results/ex2/independent-zones.json)

<a id="ex3"></a>
## Ex3 · 여러 큐브 중 지정한 큐브 놓기

여러 큐브 중 지정된 큐브의 상태를 골라 각 조작 단계의 명령을 계산합니다. 과제 채점은 97/100회·97점, 별도 놓기 평가도 97/100회입니다. [다중 큐브 정책](files/experiments/hw3/multi_cube.py) · [독립 평가](results/ex3/)

### 실행

공식 배포본의 Linux x86-64용 Python 3.12 채점기와 스터디에서 저장한 세 정책을 사용합니다. 아래 명령은 각 문제를 시드 42에서 100번 실행합니다. 출력 경로에는 새 이름을 사용하세요.

```bash
cd "$R"
python -m pytest -q tests/test_hw3.py
for n in 1 2 3; do
    python experiments/hw3/evaluate.py "$n" "study-ex$n.hwresult"
done
```

[문제별 채점 점수와 놓기 평가 시드](results/metrics.json)
