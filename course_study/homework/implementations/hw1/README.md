# HW1 · PyTorch로 이미지 분류기 만들기

네 노트북에서 텐서 연산과 학습 루프를 구현하고, MNIST에서 MLP와 TinyViT의 분류 정확도를 측정합니다. [공식 원본](../../original/hw1/) · [실행 환경 준비](../../)

<a id="ex1"></a>
## Ex1 · 텐서 연산

Attention에서는 query와 key를 특징 축을 따라 내적해 각 위치 쌍의 점수를 구합니다. Causal mask로 뒤쪽 위치를 가리면 현재 위치와 그 앞의 key에만 attention 가중치가 남습니다. `ex1.ipynb`에서 reshape, broadcasting, gather, 마스크와 softmax를 구현했습니다. 서로 다른 입력 100개에서 함수의 출력이 기준 연산 결과와 일치했습니다. [풀이 노트북](src/hw1_pytorch_tutorial/src/ex1.ipynb) · [입력별 검사 결과](results/ex1.json)

<a id="ex2"></a>
## Ex2 · 미분에서 optimizer까지

자동미분으로 얻은 gradient가 어떤 파라미터에 쌓이는지 확인하고, 길이가 다른 시퀀스를 한 배치로 묶을 때 padding 위치를 기록합니다. AdamW는 gradient의 이동평균으로 업데이트 크기를 조절하고 가중치 감쇠를 별도로 적용합니다. 입력 100개로 데이터 로딩, 파라미터 초기화, 업데이트를 검사했고 모두 통과했습니다. [풀이 노트북](src/hw1_pytorch_tutorial/src/ex2.ipynb) · [입력별 검사 결과](results/ex2.json)

<a id="ex3"></a>
## Ex3 · MNIST 분류

`Linear`, normalization, dropout과 cross-entropy를 구현해 MLP를 학습했습니다. 분류기는 이미지마다 점수가 가장 높은 숫자를 예측합니다. MNIST 시험 이미지 **10,000장 중 9,694장(96.94%)**을 맞혔습니다. [풀이 노트북](src/hw1_pytorch_tutorial/src/ex3.ipynb) · [MNIST 평가 결과](results/metrics.json)

<a id="ex4"></a>
## Ex4 · TinyViT

이미지를 패치로 나누고 위치 임베딩을 더하면, self-attention이 서로 다른 위치의 정보를 결합할 수 있습니다. 각 블록에서 특징을 변환하는 부분만 FFN, GEGLU, SwiGLU로 바꾸어 비교했습니다. 같은 48,000장으로 12 epoch씩 학습하고 검증용 5,000장의 정확도로 checkpoint를 골랐습니다. 학습 시드는 0·1·2, 시험 이미지는 매번 같은 10,000장입니다.

| 블록 | 세 시드의 시험 정확도 |
|---|---|
| FFN | 97.69%, 97.63%, 97.79% |
| GEGLU | 98.16%, 97.95%, 97.85% |
| SwiGLU | 98.32%, 98.14%, 97.75% |

[TinyViT 풀이 노트북](src/hw1_pytorch_tutorial/src/ex4.ipynb) · [세 구조의 학습 코드](src/experiments/hw1/train_variants.py) · [시드별 학습 곡선](results/tinyvit/)

### 실행

`R`을 설정한 터미널에서 실행하세요. Ex1과 Ex2는 각각 입력 100개를 모두 통과해야 합니다. 입력별 결과는 `functional-check/ex1.json`과 `functional-check/ex2.json`에 저장됩니다. 분류 노트북은 실행용 사본의 `hw1_pytorch_tutorial/src/`에서 열 수 있습니다.

```bash
cd "$R"
python -m pytest -q tests/test_hw1.py
mkdir functional-check && cd functional-check
python "$R/experiments/hw1/evaluate_functions.py"
```
