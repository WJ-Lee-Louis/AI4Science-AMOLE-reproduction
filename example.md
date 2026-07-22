# S2P loss 계산 예시

이 문서는 실제 global batch 30 대신 **batch 3**과 짧은 4-bit fingerprint를
사용해 S2P loss를 계산하는 과정을 보여준다. 실제 학습에서는 같은 계산이
1024-bit Morgan fingerprint와 global batch 30에 대해 수행된다.

숫자를 읽기 쉽게 만들기 위해 이 예시에서는 target temperature와 prediction
temperature를 모두 `1`로 둔다. 실제 실험 설정은 모두 `0.1`이므로 실제
softmax 분포는 이 예시보다 훨씬 뾰족해진다.

## 1. Batch와 augmentation

세 개의 원본 molecule-text pair가 있다고 하자.

| index | 원본 분자 | 원본 text | augmentation 결과 |
| ---: | --- | --- | --- |
| 1 | molecule 1 | text 1 | 원본 유지 |
| 2 | molecule 2 | text 2 | 유사 분자 2′로 교체 |
| 3 | molecule 3 | text 3 | 유사 분자 3′로 교체 |

Text는 교체되지 않는다. Molecule encoder에는 augmentation 이후 분자가
들어가지만, pseudo-label 계산을 위해 원본과 augmentation 이후 fingerprint를
모두 보존한다.

## 2. Tanimoto pseudo-label: Text → Molecule

1024-bit fingerprint 대신 다음과 같은 4-bit fingerprint를 사용하자.

```text
원본 fingerprint
O1 = [1, 1, 0, 0]
O2 = [0, 0, 1, 1]
O3 = [1, 0, 1, 0]

augmentation 이후 fingerprint
A1 = [1, 1, 0, 0]
A2 = [0, 0, 1, 0]
A3 = [1, 0, 0, 1]
```

Tanimoto similarity는 다음과 같다.

```text
Tanimoto(X, Y) = 두 fingerprint에서 공통으로 켜진 bit 수
                 -------------------------------------------
                 두 fingerprint 중 하나라도 켜진 bit 수
```

Text → Molecule 방향에서는 각 text에 대응하는 **원본 fingerprint**를
행으로, 비교 대상인 **augmentation 이후 fingerprint**를 열로 둔다.

```text
                       molecule target
                   A1       A2       A3
text 1 / O1       1.000    0.000    0.333
text 2 / O2       0.000    0.500    0.333
text 3 / O3       0.333    0.500    0.333
```

따라서 Tanimoto 행렬은 다음과 같다.

```text
             ┌                     ┐
             │ 1.000  0.000  0.333 │
T_text→mol = │ 0.000  0.500  0.333 │
             │ 0.333  0.500  0.333 │
             └                     ┘
```

각 행에 softmax를 적용한다. 첫 번째 행의 예시는 다음과 같다.

```text
softmax([1.000, 0.000, 0.333])
= [0.532, 0.196, 0.273]
```

전체 Text → Molecule soft target은 다음과 같다.

```text
                 molecule target 확률
             ┌                     ┐
             │ 0.532  0.196  0.273 │
Q_text→mol = │ 0.247  0.408  0.345 │
             │ 0.314  0.371  0.314 │
             └                     ┘
```

각 행의 합은 `1`이다. 예를 들어 첫 번째 행은 `text 1`이 세 molecule
target 각각에 어느 정도 대응되어야 하는지를 나타낸다.

## 3. Tanimoto pseudo-label: Molecule → Text

반대 방향에서는 augmentation 이후 fingerprint를 행으로, 각 text에 대응하는
원본 fingerprint를 열로 둔다.

```text
                      text target
                   O1       O2       O3
molecule 1 / A1   1.000    0.000    0.333
molecule 2 / A2   0.000    0.500    0.500
molecule 3 / A3   0.333    0.333    0.333
```

Tanimoto similarity가 대칭이므로 이 행렬은 앞 행렬의 transpose다.

```text
T_mol→text = T_text→molᵀ

             ┌                     ┐
             │ 1.000  0.000  0.333 │
             │ 0.000  0.500  0.500 │
             │ 0.333  0.333  0.333 │
             └                     ┘
```

하지만 transpose한 뒤 다시 **행별 softmax**를 적용하므로 최종 target 확률은
Text → Molecule target과 동일하지 않다.

```text
                 text target 확률
             ┌                     ┐
             │ 0.532  0.196  0.273 │
Q_mol→text = │ 0.233  0.384  0.384 │
             │ 0.333  0.333  0.333 │
             └                     ┘
```

대각 원소의 Tanimoto 값 자체는 양방향에서 같다. 그러나 각 행에서 함께
정규화되는 나머지 두 값이 다르므로 softmax 이후 대각 확률은 달라질 수 있다.

## 4. Model prediction: Text → Molecule

설명을 쉽게 하기 위해 encoder와 projection head를 거친 뒤 다음과 같은
2차원 **단위 벡터**가 나왔다고 하자.

```text
text embedding
t1 = [1.000, 0.000]
t2 = [0.000, 1.000]
t3 = [0.707, 0.707]

molecule embedding
m1 = [0.800, 0.600]
m2 = [0.000, 1.000]
m3 = [1.000, 0.000]
```

단위 벡터이므로 dot product와 cosine similarity가 같다.

```text
S_text→mol = Text × Moleculeᵀ

             ┌                     ┐
             │ 0.800  0.000  1.000 │  ← text 1과 molecule 1, 2, 3
             │ 0.600  1.000  0.000 │  ← text 2와 molecule 1, 2, 3
             │ 0.990  0.707  0.707 │  ← text 3과 molecule 1, 2, 3
             └                     ┘
```

각 행에 softmax를 적용하면 model prediction이 된다.

```text
             ┌                     ┐
P_text→mol = │ 0.374  0.168  0.457 │
             │ 0.329  0.491  0.180 │
             │ 0.399  0.301  0.301 │
             └                     ┘
```

예를 들어 첫 번째 행에서 model은 `text 1`이 `molecule 3`과 가장 잘
대응한다고 예측했지만, Tanimoto soft target은 `molecule 1`에 가장 높은
확률을 부여했다. 이 차이가 loss를 만든다.

## 5. Model prediction: Molecule → Text

반대 방향의 cosine similarity 행렬은 앞 prediction 행렬의 transpose다.

```text
S_mol→text = Molecule × Textᵀ = S_text→molᵀ

             ┌                     ┐
             │ 0.800  0.600  0.990 │  ← molecule 1과 text 1, 2, 3
             │ 0.000  1.000  0.707 │  ← molecule 2와 text 1, 2, 3
             │ 1.000  0.000  0.707 │  ← molecule 3과 text 1, 2, 3
             └                     ┘
```

이 행렬에 다시 행별 softmax를 적용한다.

```text
             ┌                     ┐
P_mol→text = │ 0.330  0.270  0.399 │
             │ 0.174  0.473  0.353 │
             │ 0.473  0.174  0.353 │
             └                     ┘
```

Similarity 행렬은 transpose 관계지만, 방향별로 행 softmax를 다시 하므로
prediction 확률도 단순히 서로의 transpose가 아니다.

## 6. 방향별 soft cross-entropy

각 query 행에 대해 soft target `Q`와 model prediction `P` 사이의
cross-entropy를 계산한다.

```text
L행 = -Σj Qj log(Pj)
```

Text → Molecule 첫 번째 행의 예시는 다음과 같다.

```text
Q = [0.532, 0.196, 0.273]
P = [0.374, 0.168, 0.457]

L1 = -(0.532 log 0.374
     + 0.196 log 0.168
     + 0.273 log 0.457)
   ≈ 1.084
```

세 행의 loss와 평균은 다음과 같다.

```text
Text → Molecule 행별 loss = [1.084, 1.156, 1.113]
L_text→mol = 평균 = 1.118

Molecule → Text 행별 loss = [1.095, 1.094, 1.180]
L_mol→text = 평균 = 1.123
```

최종 S2P loss는 두 방향 loss의 평균이다.

```text
L_S2P = (L_text→mol + L_mol→text) / 2
      = (1.118 + 1.123) / 2
      ≈ 1.120
```

이 scalar loss에 ER loss를 더한 뒤 optimizer update가 한 번 일어난다.

```text
L_total = L_S2P + alpha × L_ER
```

## 7. 실제 global batch 30으로 확장하면

실제 학습에서는 각 GPU가 local query 10개를 담당하고, 세 GPU에서 모은
global target 30개와 비교한다.

```text
각 GPU에서 계산: 10 × 30 logits
세 GPU 전체 효과: 30 × 30 logits

Text → Molecule: 30개 행 loss의 평균
Molecule → Text: 30개 행 loss의 평균
최종 S2P: 두 방향 평균
optimizer update: batch당 한 번
```

## 8. 현재 AMOLE 코드에 관한 중요한 주의점

위 prediction 예시는 요청에 맞춰 unit-normalized embedding의 **cosine
similarity**로 설명했다. 그러나 현재 repository의 실제 training 코드는
`F.normalize`를 적용하지 않고 다음 dot product를 사용한다.

```python
logits = torch.mm(X, Y.transpose(1, 0))
```

따라서 실제 구현을 정확히 표현하면 다음과 같다.

```text
현재 training prediction = raw embedding dot product / prediction temperature
cosine prediction         = normalized embedding dot product / temperature
```

두 방식은 embedding이 단위 벡터로 정규화되어 있을 때만 동일하다. 현재
실험 결과를 해석하거나 논문 구현을 수정할 때 이 차이를 구분해야 한다.
