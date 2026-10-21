# MatSci Lens: Interactive exploration of superconductivity prediction

## Introduction
Predicting material properties can help researchers screen candidates, but model scores alone do not show where predictions are reliable. MatSci Lens is an interactive tool for exploring machine-learning predictions of superconducting critical temperature using the public UCI Superconductivity dataset.

## Methods
The application loads the dataset, converts features and target to numeric values, removes rows with invalid targets, and evaluates a median baseline, Ridge Regression, and Extra Trees. It uses a reproducible held-out test split and reports MAE, RMSE, and R². The app visualizes measured-versus-predicted values and summarizes absolute errors.

## Results
**Replace this section after running the real dataset.** Record rows/features after cleaning, test fraction, random seed, and each model's MAE, RMSE, and R². Do not use synthetic unit-test values as scientific results. Explain which model performed best on the held-out split and note error patterns.

## Conclusions
The tool makes model comparison and errors inspectable. Results apply to the evaluated dataset and split; they do not prove causal relationships, guarantee performance on unseen material families, or replace experimental measurements.

## Future directions
Evaluate performance across material subgroups, add a robust uncertainty calibration procedure, and test performance under data shifts and alternative split strategies.
