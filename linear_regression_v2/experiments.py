import numpy as np

from model import (
    LinearRegression,
)


x = np.array(
    [
        30,
        40,
        50,
        60,
    ]
)

y = np.array(
    [
        100,
        120,
        140,
        160,
    ]
)


rates = [
    0.00001,
    0.0001,
    0.01,
]


for lr in rates:

    model = (
        LinearRegression(
            learning_rate=lr
        )
    )

    model.train(
        x,
        y,
        epochs=500,
    )

    print()

    print(
        "lr=",
        lr
    )

    print(
        "final loss=",
        round(
            model.loss_history[
                -1
            ],
            2,
        ),
    )

    print(
        "w=",
        round(
            model.w,
            2,
        )
    )