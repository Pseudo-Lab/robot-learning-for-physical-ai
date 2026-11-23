# HW3 · 시연으로 큐브 집어 옮기기

로봇은 팔 끝, 그리퍼, 큐브와 목표의 상태를 보고 팔을 어디로 움직일지, 그리퍼를 여닫을지 정합니다. 기본 모델은 시연에서 현재 상태와 뒤이은 행동들을 짝지어 학습하므로, 한 번의 예측으로 여러 시점의 행동을 출력합니다. 채점에는 접근·집기·놓기 등 시연 단계별로 다음 행동을 학습한 policy를 사용했습니다. [공식 원본](../../original/hw3/) · [기본 모델 구현](src/hw3_imitation_learning/hw3/model.py) · [실행 환경 준비](../../)

<a id="ex1"></a>
## Ex1 · 단일 큐브 놓기

Behavioral Cloning(BC)은 각 상태에서 시연자가 취한 행동을 예측하도록 policy를 학습하는 방법입니다. policy는 큐브에 접근해 집고, 들어 올린 뒤 목표에 놓습니다. 각 단계에서 로봇팔과 큐브의 상태를 보고 다음 단계로 넘어갈 시점을 정합니다. [단일 큐브 policy](src/experiments/hw3/single_cube.py) · [독립 평가](results/ex1/)

<a id="ex2"></a>
## Ex2 · DAgger와 장애물

집는 위치가 조금만 어긋나도 다음에 관측하는 큐브의 위치가 달라질 수 있습니다. DAgger(Dataset Aggregation)는 policy가 직접 방문한 상태와 그때 필요한 교정 행동을 학습 데이터에 추가합니다. 시연과 교정 데이터는 스크립트로 생성했습니다. 학습한 policy는 장애물 위치와 현재 조작 단계에 따라 행동을 선택합니다. 별도 놓기 평가에서 장애물이 왼쪽에 있을 때는 40/40회, 중앙은 12/14회, 오른쪽은 46/46회 성공했습니다. [장애물 대응 policy](src/experiments/hw3/single_cube.py) · [시연 수집 코드](src/hw3_imitation_learning/scripts/collect_scripted_demos.py) · [장애물 위치별 결과](results/ex2/independent-zones.json)

<a id="ex3"></a>
## Ex3 · 여러 큐브 중 지정한 큐브 놓기

policy는 지정된 큐브의 상태를 입력으로 삼아 각 단계에서 팔과 그리퍼에 보낼 명령을 계산합니다. [다중 큐브 policy](src/experiments/hw3/multi_cube.py) · [독립 평가](results/ex3/)

### 평가 결과

문제마다 시드를 42로 설정하고 공식 채점기로 100회 평가했습니다. 별도 평가에서는 다른 시드로 큐브를 목표에 놓았는지 확인했습니다.

| 문제 | 공식 채점 성공 횟수 | 공식 점수 | 별도 놓기 평가 성공 횟수 |
|---|---:|---:|---:|
| Ex1 · 단일 큐브 | 100/100 | 100점 | 99/100 |
| Ex2 · 장애물 | 100/100 | 100점 | 98/100 |
| Ex3 · 지정한 큐브 | 97/100 | 97점 | 97/100 |

### 실행

저장된 policy 세 개를 공식 배포본의 Linux x86-64용 Python 3.12 채점기로 평가합니다. 결과는 문제마다 `.hwresult` 파일로 저장되며, 출력 파일명은 새 이름이어야 합니다.

```bash
cd "$R"
python -m pytest -q tests/test_hw3.py
for n in 1 2 3; do
    python experiments/hw3/evaluate.py "$n" "study-ex$n.hwresult"
done
```

[문제별 채점 점수와 놓기 평가 시드](results/metrics.json)
