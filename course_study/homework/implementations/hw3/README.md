# HW3 · 시연으로 큐브 집어 옮기기

로봇은 팔 끝, 그리퍼, 큐브와 목표의 상태를 보고 팔을 어디로 움직일지, 그리퍼를 여닫을지 정합니다. 기본 모델은 시연에서 뽑은 상태와 이후의 행동 구간을 짝지어 학습합니다. 현재 상태를 입력하면 여러 시점의 행동을 한 번에 예측합니다. 아래 점수는 접근·집기·놓기 등 시연 단계별로 다음 행동을 학습한 policy로 얻었습니다. [공식 원본](../../original/hw3/) · [기본 모델 구현](src/hw3_imitation_learning/hw3/model.py) · [실행 환경 준비](../../)

<a id="ex1"></a>
## Ex1 · 단일 큐브 놓기

Behavioral Cloning(BC)은 각 상태에서 시연자가 취한 행동을 예측하도록 policy를 학습하는 방법입니다. 학습한 policy는 접근·집기·들어 올리기·놓기를 이어 가면서 현재 관측으로 다음 단계로 넘어갈 시점을 정합니다. 과제 제공 채점기로 100회 모두 성공해 100점을 받았습니다. 별도 시드에서 큐브가 실제로 놓였는지 검사한 결과는 99/100입니다. [단일 큐브 policy](src/experiments/hw3/single_cube.py) · [독립 평가](results/ex1/)

<a id="ex2"></a>
## Ex2 · DAgger와 장애물

집는 위치가 조금만 어긋나도 다음에 관측하는 큐브의 위치가 달라질 수 있습니다. DAgger(Dataset Aggregation)는 policy가 직접 방문한 상태와 그때 필요한 교정 행동을 학습 데이터에 추가합니다. 시연과 교정 데이터는 스크립트로 생성했습니다. 학습한 policy는 장애물 위치와 현재 조작 단계에 따라 행동을 선택합니다. 과제 채점에서는 100회 모두 성공해 100점을 받았습니다. 별도 놓기 평가에서는 **100회 중 98회** 성공했습니다. 장애물이 왼쪽이면 40/40, 중앙이면 12/14, 오른쪽이면 46/46회 성공했습니다. [장애물 대응 policy](src/experiments/hw3/single_cube.py) · [시연 수집 코드](src/hw3_imitation_learning/scripts/collect_scripted_demos.py) · [장애물 위치별 결과](results/ex2/independent-zones.json)

<a id="ex3"></a>
## Ex3 · 여러 큐브 중 지정한 큐브 놓기

여러 큐브 중 지정된 큐브의 상태를 골라 각 조작 단계의 명령을 계산합니다. 과제 채점에서는 100회 중 97회 성공해 97점을 받았습니다. 별도 놓기 평가에서도 100회 중 97회 성공했습니다. [다중 큐브 policy](src/experiments/hw3/multi_cube.py) · [독립 평가](results/ex3/)

### 실행

공식 배포본의 Linux x86-64용 Python 3.12 채점기로 저장된 policy 세 개를 평가합니다. 각 문제는 시드 42에서 시작해 100회 실행합니다. 출력 경로에는 새 이름을 사용하세요.

```bash
cd "$R"
python -m pytest -q tests/test_hw3.py
for n in 1 2 3; do
    python experiments/hw3/evaluate.py "$n" "study-ex$n.hwresult"
done
```

[문제별 채점 점수와 놓기 평가 시드](results/metrics.json)
