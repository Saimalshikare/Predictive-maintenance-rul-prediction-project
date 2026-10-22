# Model Comparison - FD001

Validation = 20 held-out engines (split by engine, random_state=42).

| Model | RMSE | MAE | NASA |
|---|---|---|---|
| Extra Trees | 15.52 | 11.12 | 21,816.99 |
| Gradient Boosting | 15.73 | 11.80 | 21,603.47 |
| Random Forest | 16.81 | 11.75 | 31,476.51 |
| Linear Regression | 18.03 | 14.07 | 21,524.00 |
| Ridge | 18.03 | 14.07 | 21,522.96 |
| Baseline (Mean) | 41.72 | 36.93 | 1,285,855.81 |

**Selected model:** Extra Trees (lowest RMSE and MAE on validation).
The gap between the top two models is small.

## Official test set (100 engines, one prediction per engine)

| RMSE | MAE | R2 | NASA |
|---|---|---|---|
| 17.87 | 13.12 | 0.815 | 747.6 |
