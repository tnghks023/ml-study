import numpy as np

from model import (
    LinearRegression
)

# 면적
x = np.array(
    [
        30,40,50,60,
    ]
)

# 가격
y = np.array(
    [
        100, 120,140,160
    ]
)

model = (
    LinearRegression(
        learning_rate=0.0001
    )
)

model.train(
    x,y,epochs=5000,
)

print()

print(
    "w:", round(model.w, 2)
)

print("b: ", round(model.b, 2))

test = 45

print(
    f"{test} predict :",
    round(
        model.predict(
            test
        ),
        2,
    )
)