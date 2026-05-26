# Linear Regression + Gradient Descent 정리

## 목표

선형 회귀(Linear Regression)를 직접 구현하면서

- Prediction
- Loss Function
- Gradient
- Gradient Descent

가 어떻게 연결되는지 이해한다.

---

# 전체 흐름

```text
Input(x)
↓

Prediction(y_hat)

↓

Loss 계산

↓

Gradient 계산

↓

Weight 업데이트

↓

반복
```

---

# 1. Prediction (예측)

모델 식:

\[
\hat y = wx+b
\]

용어:

| 기호  | 의미                |
| ----- | ------------------- |
| x     | 입력값 (Input)      |
| y     | 실제값 (Label)      |
| y_hat | 예측값 (Prediction) |
| w     | 가중치 (Weight)     |
| b     | 편향 (Bias)         |

예시:

\[
x=50
\]

\[
w=2
\]

\[
b=10
\]

예측:

\[
\hat y=2\times50+10=110
\]

---

# 2. Loss Function

Loss Function:

예측이 얼마나 틀렸는지 계산.

이번 실습:

MSE (Mean Squared Error)

\[
L=
\frac1n
\sum
(y-\hat y)^2
\]

코드:

```python
loss = np.mean(
    (
        y-pred
    )**2
)
```

설명:

```text
실제값 - 예측값
↓

제곱
↓

평균
```

---

# 3. Gradient 계산

Loss:

\[
L=
(y-(wx+b))^2
\]

치환:

\[
u=y-(wx+b)
\]

그러면

\[
L=u^2
\]

---

## 바깥 미분

\[
\frac{dL}{du}=2u
\]

---

## 안쪽 미분

\[
\frac{du}{dw}=-x
\]

---

## Chain Rule (연쇄법칙)

\[
\frac{dL}{dw}
=
\frac{dL}{du}
\times
\frac{du}{dw}
\]

대입:

# \[

2u(-x)
\]

\[
=-2xu
\]

다시 치환:

\[
=-2x(y-\hat y)
\]

전체 평균:

\[
dw=
-2
\cdot
mean(
x(y-\hat y)
)
\]

코드:

```python
dw = (
    -2
    *
    np.mean(
        x
        *
        (
            y-pred
        )
    )
)
```

---

## b 미분

\[
db=
-2
\cdot
mean(
y-\hat y
)
\]

코드:

```python
db = (
    -2
    *
    np.mean(
        y-pred
    )
)
```

---

# 4. Gradient Descent

공식:

\[
w=
w-
\alpha
\frac{\partial L}{\partial w}
\]

\[
b=
b-
\alpha
\frac{\partial L}{\partial b}
\]

설명:

```text
현재 위치
↓

gradient 계산

↓

반대 방향 이동

↓

loss 감소
```

코드:

```python
self.w -= (
    self.lr
    *
    dw
)

self.b -= (
    self.lr
    *
    db
)
```

용어:

| 기호 | 의미          |
| ---- | ------------- |
| lr   | Learning Rate |
| dw   | w 방향 기울기 |
| db   | b 방향 기울기 |

---

# 왜 빼기(-)를 사용할까?

Gradient:

```text
Loss 증가 방향
```

우리는:

```text
Loss 감소 방향
```

으로 이동해야 함.

그래서:

\[
-\nabla L
\]

사용.

---

# 최종 학습 흐름

```text
x

↓

predict()

↓

y_hat

↓

loss()

↓

gradient

↓

update()

↓

repeat
```

---

# 내가 이해한 핵심

- y = 실제값
- y_hat = 예측값
- Loss = 오차 계산
- Gradient = 어느 방향으로 움직일지
- Learning Rate = 얼마나 움직일지
- Gradient Descent = 실제 이동

---
