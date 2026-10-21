# CrystalCast: Predicting crystallization onset in metallic glasses

## Introduction
Metallic glasses are amorphous alloys that can begin to crystallize when heated. Their crystallization-onset temperature (Tx) is an important thermal characteristic. CrystalCast asks whether alloy composition alone can estimate Tx for iron-based metallic glasses.

## Methods
We use the open RWTH Aachen dataset of iron-based metallic-glass alloys. Composition is used as the predictor; measured crystallization, glass-transition, and liquidus temperatures are excluded from the feature set to avoid target leakage. A median baseline, Ridge Regression, and Extra Trees are evaluated with a reproducible held-out split. The app reports MAE, RMSE, and R², visualizes measured versus predicted Tx, and accepts a user-entered composition for exploratory prediction.

## Results
**Complete this section only after running the app on the real dataset.** Record the number of rows with valid Tx, the number of composition features, the test fraction, random seed, and each model's actual MAE, RMSE, and R². State which model performed best on the selected split and discuss visible error patterns. Do not use synthetic unit-test numbers as scientific results.

## Conclusions
The prototype makes the relationship between alloy composition and predicted Tx easier to explore. The results apply to the dataset and split evaluated; they do not prove causal relationships or replace thermal experiments. Composition-only predictions may miss effects from heating rate, processing history, and differences between alloy families.

## Future directions
Use grouped or composition-family splits to better test generalization, add uncertainty estimates, and incorporate experimental conditions when sufficiently complete records are available.
