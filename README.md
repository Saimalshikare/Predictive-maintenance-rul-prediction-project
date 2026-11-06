# Predictive Maintenance: Remaining Useful Life (RUL) Prediction

## 1. The Problem

An aircraft engine has to be repaired at the right time. If it is repaired too early, money and engine life are wasted. If it is repaired too late, the engine can fail in service, which is dangerous. So the question every airline asks is: how many more flights can this engine safely do?

That number is called the Remaining Useful Life (RUL). In this project I predict it from the sensor readings that the engine records during every flight.

## 2. The Data

I used the NASA C-MAPSS turbofan engine dataset (FD001 part), which is simulated run-to-failure data.

Source: https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data

The paper behind the data: Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08.

**Where to load it:** the data is not stored in this repository. Download the dataset and copy these three files into the folder `data/raw/`:

```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

What is inside:

- `train_FD001.txt`: 100 engines that were run until they failed (20,631 rows).
- `test_FD001.txt`: 100 different engines whose records stop before failure (13,096 rows).
- `RUL_FD001.txt`: the true remaining life of each test engine, used only to check my predictions.
- Every row is one flight with 26 columns: engine id, cycle number, 3 operating settings and 21 sensor readings.
- FD001 has one operating condition and one fault type (high pressure compressor degradation).

## 3. What I Solve

I built a regression model. It takes the sensor history of an engine and returns one number: how many flights are left. The goal was to do it properly from start to end: explore the data, build features, compare models, choose one without cheating, package the code, test it, and show the result in a dashboard.

## 4. Project Structure

I followed the layout that is commonly used for production-style machine learning projects. Experiments are kept separate from the reusable code, all settings live in one file, and the code is tested.

```
configs/config.yaml        all settings (paths, window size, RUL cap, model settings)
data/raw/                  original NASA files, never edited, not in the repository
data/processed/            data prepared for modeling
notebooks/                 01_eda, 02_feature_engineering, 03_model_experiments
src/rul/
    utils/                 config reader and logger
    data/                  load_data.py, preprocess.py
    features/              build_features.py
    models/                train.py, evaluate.py, predict.py
    pipeline.py            runs the whole process in one command
app/streamlit_app.py       dashboard
models/                    saved model, scaler and meta.json (created by the pipeline)
reports/                   model comparison table and figures
tests/                     pytest tests
requirements.txt
.gitignore
README.md
```

Why this structure: the notebooks are where I try ideas. Once something works, I move it into `src/rul` as small functions, so the pipeline, the tests and the dashboard all use the same code. Settings such as the window size (30) and the RUL cap (125) are only written in `config.yaml`, not inside the code.

## 5. Step by Step: Problems I Faced and How I Solved Them

**Step 1: Setup.** I created the folders, `config.yaml`, `.gitignore`, `requirements.txt` and a virtual environment.

**Step 2: Understanding the data (EDA).** I looked at data types, missing values, duplicates, the life of every engine and the sensor trends.

- Problem: 7 columns never change (`setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`). A column that never changes cannot tell the model anything. Solution: I removed them. `sensor_6` has only two different values (21.61 in about 98% of rows), but it is not strictly constant, so I kept it.
- Problem: engines fail after very different numbers of flights, from 128 up to 362. Solution: I use engine-wise splitting and cap the RUL (see below).
- Problem: the sensors are very noisy. Plotting one engine shows a clear upward or downward trend hidden under a zig-zag line. Solution: rolling averages.
- The sensors most related to RUL were `sensor_11` (correlation -0.70), `sensor_4` (-0.68), `sensor_12` (+0.67) and `sensor_7` (+0.66). The settings had almost no relation to RUL in FD001.

**Step 3: Feature engineering.**

- Problem: the training data has no RUL column. Solution: for every row I calculated last cycle of the engine minus the current cycle.
- Problem: early in an engine's life the sensors look almost the same, so 200 flights left and 150 flights left cannot be told apart. Every engine ran at least 128 flights. Solution: I capped the RUL at 125.
- Problem: rows from the same engine are very similar, so a random row split would put almost the same data in both train and validation and give results that look too good. Solution: I split by engine, 80 engines for training and 20 for validation.
- Problem: scaling with the full data would leak information from validation into training. Solution: the scaler learns only from the training engines.
- Problem: single readings are noisy. Solution: for every sensor I added a 30-flight rolling mean and rolling standard deviation, calculated inside each engine, so one engine never mixes with another. That gives 15 sensors x 3 = 45 features.
- Problem: on the test set I need one prediction per engine. Solution: I use the last flight of each engine.

**Step 4: Training and comparing models.** Described in section 6.

**Step 5: Choosing the model and testing it once.** Described in section 7.

**Step 6: Moving the code into `src/rul`.** The notebook code became small functions: reading the data, checking it, scaling, features, training, evaluating, predicting. I checked that the pipeline gives the same numbers as my notebook.

- Problem: if the data is wrong, a model can silently give wrong numbers. Solution: `preprocess_data` checks for missing values, wrong ids, duplicate engine-cycle rows and missing columns, and raises a clear error. The metric functions also refuse inputs of different length or with NaN values.
- Problem: a model can break if training and prediction prepare the data differently. Solution: after training, the pipeline saves the model, the scaler and a `meta.json` with the removed columns, the sensors, the 45 feature names, the window size and the RUL cap. The prediction code reads these, so it repeats exactly the same steps.

**Step 7: Tests.** I wrote 36 pytest tests: 9 for preprocessing, 11 for features, 11 for the metrics and 5 for prediction. They check, for example, that RUL is 0 at the last cycle, that rolling features never mix two engines, that the scaler does not learn from new data, that wrong input raises an error, and that predictions stay between 0 and 125.

**Step 8: Running the project.**

1. Put the three FD001 files in `data/raw/` (section 2).
2. Install the libraries:

```
python -m pip install -r requirements.txt
```

3. Train the model and print the results (about 30 seconds). This creates the files in `models/`.

Windows PowerShell:
```
$env:PYTHONPATH="src"
python -m rul.pipeline
```

Mac / Linux:
```
export PYTHONPATH=src
python -m rul.pipeline
```

4. Run the tests:

```
python -m pytest tests -q
```

**Step 9: Dashboard.**

```
python -m streamlit run app/streamlit_app.py
```

It opens at `http://localhost:8501`. The app shows the predicted RUL of every engine and marks each one as Critical (below 20 cycles), Warning (below 50) or Healthy; both limits can be changed with sliders. With the default limits the 100 test engines are split into 13 Critical, 15 Warning and 72 Healthy. You can choose an engine and look at the trend of any sensor, and on the official test set the app compares each prediction with the true value. You can also upload your own file (same 26-column format, no header) to predict engines the model has never seen. The less history an engine has, the less reliable the prediction is, because the rolling features need several flights.

## 6. Model Training and Results of All Models

I trained every model on the same 80 training engines, with the same 45 features, and checked it on the same 20 validation engines. I used `random_state=42` everywhere so the results can be repeated, and I clipped every prediction between 0 and the RUL cap.

Models and settings:

- Baseline: always predicts the average RUL of the training data (87.0). It uses no sensors at all, so every real model has to beat it.
- Linear Regression: default settings.
- Ridge: alpha = 1.0.
- Random Forest: 200 trees, max_depth = 15, min_samples_leaf = 5.
- Extra Trees: 200 trees, min_samples_leaf = 5.
- Gradient Boosting: 200 trees, learning_rate = 0.05, max_depth = 3.

Metrics, where lower is better for all of them:

- RMSE: average error that punishes big mistakes more.
- MAE: average error in flights.
- NASA score: the PHM08 score. It punishes late predictions (saying an engine has more life than it really has) more than early ones, because the late mistake is the dangerous one.

Results on the 20 validation engines:

| Model | RMSE | MAE | NASA score |
|---|---|---|---|
| Extra Trees | 15.52 | 11.12 | 21,816.99 |
| Gradient Boosting | 15.73 | 11.80 | 21,603.47 |
| Random Forest | 16.81 | 11.75 | 31,476.51 |
| Linear Regression | 18.03 | 14.07 | 21,524.00 |
| Ridge | 18.03 | 14.07 | 21,522.96 |
| Baseline (mean) | 41.72 | 36.93 | 1,285,855.81 |

What I learned from this table:

- Every real model is far better than the baseline, so the sensor features really contain information.
- The tree models beat the straight-line models, so the relation between the sensors and the RUL is curved, not a straight line.
- Random Forest has the best-looking error but the worst NASA score among the tree models, so it makes more of the dangerous late predictions. This is why I do not choose a model from a single number.

## 7. Model Selection and the Rest of the Process

**Selection.** I chose Extra Trees because it has the lowest RMSE and the lowest MAE on the validation engines. Gradient Boosting is very close, and it has a slightly lower NASA score (21,603 against 21,817). The validation set has only 20 engines, so I do not claim that Extra Trees is clearly the best, only that it was the best in this check.

**Final training and test.** After choosing, I trained Extra Trees on all 100 training engines and tested it once on the 100 test engines, using the last flight of each engine and the true values from `RUL_FD001.txt`. I did not change anything after seeing this result.

| RMSE | MAE | R2 | NASA score |
|---|---|---|---|
| 17.87 | 13.12 | 0.815 | 747.6 |

RMSE, MAE and the NASA score are the main metrics for this problem. R2 is only a supporting number: on validation it was 0.862 (adjusted R2 0.860). On the test set the adjusted R2 drops to 0.661, but only because there are 100 samples against 45 features, which makes the adjustment very strong. The test error is a little higher than the validation error, which is normal for engines the model has never seen.

**Explaining the model.** I used SHAP to see which features the model relies on. The figures are saved in `reports/figures/` (`shap_importance.png` and `shap_summary.png`). The model comparison table is saved in `reports/model_comparison.md`.

**Saving and using it.** The pipeline saves the model, the scaler and `meta.json` in `models/`. The prediction code loads them and repeats the same preparation for any new engine data, and the dashboard uses this code. The model files are not stored in the repository, because they are created by the pipeline.

**Things to keep in mind when reading the results.**

- I trained and tested only on FD001. FD002 and FD004 have 6 operating conditions, which my scaling does not handle yet, so the pipeline prints a warning for them. I did not run FD003.
- Extra Trees fits the training engines much more closely than new engines, so there is some overfitting. I did not tune any hyperparameters.
- Because the RUL is capped at 125, an engine that really has more than 125 flights left cannot be predicted correctly.
- The data is simulated, so the model should not be used on real engines without retraining.

## 8. Ideas for Improvement

1. Use cross-validation split by engine (for example 5 folds), so the model choice does not depend on only 20 validation engines.
2. Tune the hyperparameters (`min_samples_leaf`, `max_depth`, `max_features`) with that cross-validation, to reduce the overfitting.
3. Scale the sensors separately for each operating condition, then run FD002, FD003 and FD004 and compare all four datasets.
4. Try a sequence model such as an LSTM or a 1D CNN and compare it with these results.
5. Give a range instead of one number. The 200 trees of the Extra Trees model disagree with each other, and that spread could be shown as an uncertainty range for each engine.
6. Make the model careful about late predictions. Since the NASA score punishes them, I could train with a loss that punishes late errors more, or add a safety margin to the predicted RUL, and check what it does to the score.
7. Put the dashboard online with a small sample file, so someone can try it without downloading the full dataset. The saved model is large, so it would need to be compressed first.
8. Use the dashboard alerts to prioritise engines: sort by predicted RUL and also by how uncertain the prediction is.
