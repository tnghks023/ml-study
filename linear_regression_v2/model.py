import numpy as np


class LinearRegression:

    def __init__(
        self,
        learning_rate: float,
    ):
        self.w = 0.0
        self.b = 0.0

        self.lr = learning_rate

        self.loss_history = []

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        return (
            self.w * x
            + self.b
        )

    def loss(
        self,
        x,
        y,
    ):

        pred = self.predict(
            x
        )

        return np.mean(
            (
                y
                -
                pred
            ) ** 2
        )

    def train(
        self,
        x,
        y,
        epochs=500,
    ):

        for epoch in range(
            epochs
        ):

            pred = (
                self.predict(
                    x
                )
            )

            dw = (
                -2
                *
                np.mean(
                    x
                    *
                    (
                        y
                        -
                        pred
                    )
                )
            )

            db = (
                -2
                *
                np.mean(
                    y
                    -
                    pred
                )
            )

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

            current_loss = (
                self.loss(
                    x,
                    y,
                )
            )

            self.loss_history.append(
                current_loss
            )