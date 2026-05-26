import numpy as np

class LinearRegression:

    def __init__(
        self,
        learning_rate: float = 0.001,
    ):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate

    
    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        
        return (
            self.w * x
            + self.b
        )
    
    def compute_loss(
        self,
        x,
        y,
    ):
        
        pred = self.predict(x)

        return np.mean((y - pred) ** 2)
    
    def train(
        self,
        x,
        y,
        epochs=1000,
    ): 
        n = len(x)

        for epoch in range(
            epochs
        ):
            pred = self.predict(x)

            dw = (
                -2 * np.mean(x*(y - pred))
            )
            
            db = (
                -2 * np.mean(y - pred)
            )

            self.w -= (
                self.lr * dw
            )

            self.b -= (
                self.lr * db
            )

            if(
                epoch % 100==0
            ):
                loss = (
                    self.compute_loss(x,y)
                )

                print(
                    f"epoch={epoch}"
                    f" "
                    f"loss={loss:.2f}"
                )
    
