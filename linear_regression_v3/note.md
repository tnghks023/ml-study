## Loss Curve Visualization

Compare learning rates:

- 0.00001
- 0.0001
- 0.001

Observe:

- convergence
- divergence
- training stability

그래프를 더 자세하기 보기위해 로그 스케일(Log Scale) 추가
->
기존 learning_rate가 너무커져서 수정 및 발산값 잘라내기 추가
->
Data PreProcessing(x = x / 100)
하니까 그래프가 더 명확하게 보임
Learning rate 0.001도 가능해짐
