# CrystalCast

**OSI Hackathon 2026 — Molecular and Material Property Prediction**

CrystalCast is an interactive research demo focused on one specific materials-science question:

> Can alloy composition help predict the crystallization-onset temperature (Tx) of iron-based metallic-glass alloys?

Metallic glasses lack the long-range ordered structure of crystals. When heated, they may begin to crystallize. Estimating the onset temperature could help researchers compare candidate alloys before conducting experiments. CrystalCast evaluates models against held-out published measurements and lets a user enter a composition for an exploratory estimate.

## What it does

- Loads the open RWTH Aachen dataset for 701 iron-based metallic-glass alloys.
- Uses alloy composition as the model input; it deliberately excludes measured crystallization, glass-transition, and liquidus temperatures from the features to avoid target leakage.
- Compares a median baseline, Ridge Regression, and Extra Trees.
- Reports MAE, RMSE, and R² on a reproducible held-out test split.
- Plots measured versus predicted crystallization-onset temperature and summarizes prediction errors.
- Accepts a compact composition such as Fe72B20Si8 for an exploratory prediction, if its elements are represented in the data.

## Dataset and attribution

Bobzin et al., “Data driven development of iron-based metallic glasses using artificial neural networks,” associated dataset hosted by RWTH Aachen University (2023).

- Dataset record: https://publications.rwth-aachen.de/record/969330
- Direct Excel file: https://publications.rwth-aachen.de/record/969330/files/Fe_MG_Database.xlsx
- DOI: https://doi.org/10.18154/RWTH-2023-09114

The source record describes 701 iron-based metallic-glass alloys and includes composition and, where available, crystallization temperature, glass-transition temperature, liquidus temperature, heating rate, and manufacturing information. CrystalCast uses composition to predict Tx. Please consult the source record for attribution and reuse terms.

## Project status

This is a hackathon prototype. Offline unit tests validate the data-preparation and modelling logic on synthetic test data only. The actual RWTH workbook and the full Marimo UI must be run locally before reporting scientific metrics. Do not present synthetic test outcomes as scientific results.

## Requirements

- Python 3.11 or newer recommended
- Internet access for initial dataset download and dependency installation
- Alternatively, download the Excel file from the source record and place it at data/Fe_MG_Database.xlsx

## Setup on Windows

Open this repository in VS Code. In Terminal → New Terminal, run:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

If PowerShell blocks activation, use a terminal that supports environment activation rather than changing system-wide security settings.

## Tests and run

Run the offline tests first:

    python -m unittest discover -s tests -v

Then launch the interactive notebook:

    python -m marimo edit app.py

The first dataset load needs internet access unless the workbook has been downloaded into the local data folder. The data folder is excluded from Git so the source workbook is not accidentally committed.

## Model evaluation

Use the same test fraction and random seed when comparing models. The app reports:

- MAE: average absolute prediction error, in kelvin.
- RMSE: error metric that penalizes larger errors more strongly, in kelvin.
- R²: proportion of test-target variation explained relative to a mean baseline.

Use the actual metrics displayed after loading the source dataset. A random split can overestimate performance on entirely new alloy families, especially where related compositions appear in both train and test sets. Composition-only models also omit processing history and measurement conditions.

## Two-person collaboration plan

- **Data and modelling:** verify the data columns and target, run tests, compare models, record true-data metrics, and write the limitations.
- **Interface and communication:** test the Marimo app and composition-input flow, review charts, prepare the short demo, and finish the submission text.
- Both teammates should inspect the results before submission. Commit changes in small increments and coordinate before editing the same file.

## Scientific cautions

- A predicted Tx is an estimate, not an experimental measurement.
- Feature relationships do not prove causation.
- The model is trained on iron-based metallic glasses and should not be assumed to apply to every alloy or material family.
- The estimate is not a calibrated uncertainty interval.
- Record the test fraction, random seed, sample count, and real model metrics in the write-up.

## Team

- Repository owner: @geraldchin07-arch
- Teammate: add their GitHub username through repository collaborator settings.

## License

Choose a license for this project's code before encouraging reuse. The source dataset and project code may have different terms.