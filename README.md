# MatSci Lens

**OSI Hackathon 2026 — Molecular and Material Property Prediction**

MatSci Lens is an interactive research demo for exploring superconducting materials data and evaluating machine-learning models that predict critical temperature. The aim is to make model performance, prediction errors, and scientific limitations visible—not to claim that predictions prove new materials will work experimentally.

## Project status

This repository is a starter implementation. The app and dependencies must be run locally and the real-data workflow must be checked before reporting scientific results. Do not put unverified scores in the write-up.

## Planned features

- Load the public UCI Superconductivity dataset.
- Compare a median baseline, Ridge Regression, and Extra Trees.
- Report MAE, RMSE, and R² on held-out test data.
- Visualise predicted versus measured critical temperatures and error patterns.
- Discuss model limitations and distinguish correlations from causal explanations.

## Dataset

UCI Machine Learning Repository, **Superconductivity Data**:
https://archive.ics.uci.edu/dataset/464/superconductivty%2Bdat

Dataset attribution and terms should be checked at the source before redistribution. The project should download data on demand or use a local cache; do not commit a large raw dataset to Git.

## Requirements

- Python 3.11 or newer recommended
- Internet access for initial dependency installation and dataset retrieval

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/geraldchin07-arch/YzY.git
cd YzY
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If working on the feature branch, switch to it after cloning:

```powershell
git checkout feature/matsci-lens-starter
```

## Run

Once the application files are added and dependencies installed:

```powershell
python -m unittest discover -s tests -v
python -m marimo edit app.py
```

The test command should be run first. A successful synthetic unit test run validates software behaviour on test fixtures only; it does **not** prove model performance on the scientific dataset.

## Two-person work plan

- **Data and modelling:** data loading/cleaning, model pipeline, reproducible split, metrics, and error analysis.
- **Interface and communication:** Marimo user interface, visualisations, README/testing, screenshots, one-page write-up, and demo video.

Both teammates should review the integrated app and check the reported results before submission. Use branches and pull requests rather than editing the same files concurrently.

## Validation checklist

- [ ] Install dependencies in a clean environment.
- [ ] Run unit tests and record the output.
- [ ] Launch the Marimo app.
- [ ] Verify dataset row/column counts and target column.
- [ ] Run every model option successfully.
- [ ] Save real MAE, RMSE, and R² values from the held-out test set.
- [ ] Check charts and explain limitations.
- [ ] Add screenshots and a demo only after the app works.
- [ ] Replace all placeholders in the write-up with verified information.

## Scientific cautions

- A held-out test score estimates performance for the chosen split; it is not proof of generalisation to all new materials.
- Feature importance and association do not establish causation.
- Model predictions should not be described as experimental confirmation.
- Report the split procedure, random seed, metrics, and any known limitations.

## Team

- Repository owner: @geraldchin07-arch
- Teammate: add GitHub username in the repository's collaborator settings.

## License

Choose a project license before public reuse. Dataset licensing/terms and project code licensing are separate matters.
