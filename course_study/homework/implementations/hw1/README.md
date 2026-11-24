# HW1 · PyTorch로 이미지 분류기 만들기

네 노트북에서 텐서 연산과 학습 루프를 구현하고, MNIST에서 MLP와 TinyViT의 분류 정확도를 측정합니다.

[공식 원본](../../original/hw1/) · [실행 환경 준비](../../)

<a id="ex1"></a>

## Ex1 · 텐서 연산

Attention에서는 query와 key를 특징 축을 따라 내적해 각 위치 쌍의 점수를 구합니다. Causal mask로 뒤쪽 위치를 가리면 현재 위치와 그 앞의 key에만 attention 가중치가 남습니다.

`ex1.ipynb`에서 reshape, broadcasting, gather, 마스크와 softmax를 구현했습니다. 서로 다른 입력 100개에서 함수의 출력이 기준 연산 결과와 일치했습니다.

[풀이 노트북](src/hw1_pytorch_tutorial/src/ex1.ipynb) · [입력별 검사 결과](results/ex1.json)

<a id="ex2"></a>

## Ex2 · 미분에서 optimizer까지

자동미분으로 얻은 gradient가 어떤 파라미터에 쌓이는지 확인하고, 길이가 다른 시퀀스를 한 배치로 묶을 때 padding 위치를 기록합니다. AdamW는 gradient의 이동평균으로 업데이트 크기를 조절하고 가중치 감쇠를 별도로 적용합니다.

입력 100개로 데이터 로딩, 파라미터 초기화, 업데이트를 검사했고 모두 통과했습니다.

[풀이 노트북](src/hw1_pytorch_tutorial/src/ex2.ipynb) · [입력별 검사 결과](results/ex2.json)

<details>
<summary>Ex1·Ex2 연산 예제 보기</summary>

왼쪽은 네 위치의 causal attention 가중치를 한 행씩 보여 줍니다. 미래 위치의 가중치는 0입니다. 오른쪽은 손실 $(w-1)^2$를 줄이도록 AdamW로 $w$를 20번 갱신한 결과입니다.

![미래 위치를 가리는 attention 행렬과 AdamW의 파라미터 갱신](media/operations.gif)

</details>

<a id="ex3"></a>

## Ex3 · MNIST 분류

`Linear`, normalization, dropout과 cross-entropy를 구현해 MLP를 학습했습니다. 분류기는 이미지마다 점수가 가장 높은 숫자를 예측합니다.

MNIST 시험 이미지 **10,000장 중 9,694장(96.94%)**을 맞혔습니다.

[풀이 노트북](src/hw1_pytorch_tutorial/src/ex3.ipynb) · [MNIST 평가 결과](results/metrics.json)

<a id="demo"></a>

### 학습하면서 숫자 예측이 어떻게 달라질까

시연을 위해 노트북의 MLP를 시드 0으로 다시 학습했습니다. MNIST 시험 세트에서 숫자별로 처음 등장하는 한 장씩을 골라, 매 epoch 같은 열에 놓았습니다. 이미지 아래에는 정답, 모델의 예측, 해당 예측의 확률이 차례로 나옵니다. 상단의 정확도는 시험 이미지 10,000장 전체에서 측정한 값입니다.

![MLP가 0~5 epoch 동안 같은 손글씨 열 장에 내린 예측](media/mlp.gif)

두 분류 GIF에는 Yann LeCun과 Corinna Cortes의 MNIST 손글씨를 확대해 모델의 예측값을 덧붙였습니다. 두 GIF의 이용 조건은 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)입니다. [MNIST 데이터와 저작권 표기](https://keras.io/api/datasets/mnist/)

<a id="ex4"></a>

## Ex4 · TinyViT

이미지를 패치로 나누고 위치 임베딩을 더하면, self-attention이 서로 다른 위치의 정보를 결합할 수 있습니다. 각 블록에서 특징을 변환하는 부분만 FFN, GEGLU, SwiGLU로 바꾸어 비교했습니다.

같은 48,000장으로 12 epoch씩 학습하고 검증용 5,000장의 정확도로 checkpoint를 골랐습니다. 학습 시드는 0·1·2, 시험 이미지는 매번 같은 10,000장입니다.

| 블록 | 세 시드의 시험 정확도 |
|---|---|
| FFN | 97.69%, 97.63%, 97.79% |
| GEGLU | 98.16%, 97.95%, 97.85% |
| SwiGLU | 98.32%, 98.14%, 97.75% |

[TinyViT 풀이 노트북](src/hw1_pytorch_tutorial/src/ex4.ipynb) · [세 구조의 학습 코드](src/experiments/hw1/train_variants.py) · [시드별 학습 곡선](results/tinyvit/)

<details>
<summary>TinyViT의 epoch별 예측 보기</summary>

FFN을 쓰는 TinyViT를 시드 0으로 12 epoch 다시 학습했습니다. MLP와 같은 손글씨를 보여 줍니다. 위 표는 기존 실험에서 검증 정확도로 고른 모델의 시험 결과입니다.

![TinyViT의 학습 전부터 12 epoch까지 같은 숫자에 대한 예측](media/tinyvit.gif)

</details>

## 실행

`R`을 설정한 터미널에서 실행하세요. Ex1과 Ex2는 각각 입력 100개를 모두 통과해야 합니다. 입력별 결과는 `functional-check/ex1.json`과 `functional-check/ex2.json`에 저장됩니다. 분류 노트북은 실행용 사본의 `hw1_pytorch_tutorial/src/`에서 열 수 있습니다.

```bash
cd "$R"
python -m pytest -q tests/test_hw1.py
mkdir functional-check && cd functional-check
python "$R/experiments/hw1/evaluate_functions.py"
```
