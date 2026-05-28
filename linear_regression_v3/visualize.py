
import numpy as np
import matplotlib.pyplot as plt

from model import (
    LinearRegression,
)

x = np.array(
    [30, 40, 50, 60]
)

y = np.array(
    [100, 120, 140, 160]
)

rates = [
    0.00001,
    0.0001,
    0.001,
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
        epochs=300,
    )

    plt.plot(
        model.loss_history,
        label=f"lr={lr}"
    )

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Loss Curve"
)

plt.legend()

plt.show()