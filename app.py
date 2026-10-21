"""CrystalCast: interactive crystallization-onset temperature prediction."""
import marimo
app = marimo.App(width="medium")

@app.cell
def _():
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    from src.matsci_lens import evaluate_model, load_crystallization_dataset, predict_tx
    return evaluate_model, load_crystallization_dataset, mo, np, plt, predict_tx

@app.cell
def _(mo):
    mo.md(r"""
    # CrystalCast
    ### Predicting crystallization onset temperature in iron-based metallic glasses

    Metallic glasses lack the long-range ordered structure of crystals. When heated, they can begin to crystallize. CrystalCast explores whether alloy composition alone can help estimate the crystallization-onset temperature, **Tx (K)**.

    This is a screening and learning tool—not an experimental measurement or proof that an alloy will behave as predicted.
    """)
    return

@app.cell
def _(mo):
    load_button = mo.ui.run_button(label="Load RWTH iron-based metallic-glass dataset")
    load_button
    return (load_button,)

@app.cell
def _(load_button, load_crystallization_dataset, mo):
    if load_button.value:
        try:
            X, y = load_crystallization_dataset()
            load_message = mo.md(
                f"**Dataset ready:** {len(X):,} rows with measured Tx · {X.shape[1]} composition features. "
                f"Observed Tx range: {y.min():.1f}–{y.max():.1f} K."
            )
        except Exception as exc:
            X, y = None, None
            load_message = mo.md(f"**Dataset load failed:** {type(exc).__name__}: {exc}")
    else:
        X, y = None, None
        load_message = mo.md("Click the button to download and prepare the open RWTH dataset. Internet access is needed on the first run unless you place the XLSX in data/Fe_MG_Database.xlsx.")
    load_message
    return X, y

@app.cell
def _(X, mo, y):
    if X is not None and y is not None:
        model_choice = mo.ui.dropdown(
            options=["Median Baseline", "Ridge Regression", "Extra Trees"],
            value="Extra Trees", label="Model for evaluation and prediction"
        )
        test_fraction = mo.ui.slider(start=0.1, stop=0.3, step=0.05, value=0.2, label="Held-out test fraction")
        seed = mo.ui.number(value=42, start=0, stop=10000, label="Random seed")
        mo.vstack([model_choice, test_fraction, seed])
    else:
        model_choice, test_fraction, seed = None, None, None
    return model_choice, seed, test_fraction

@app.cell
def _(X, evaluate_model, mo, model_choice, seed, test_fraction, y):
    if X is not None and model_choice is not None:
        try:
            evaluation = evaluate_model(X, y, model_choice.value, random_state=int(seed.value), test_size=float(test_fraction.value))
            metrics = mo.md(f"""## Held-out evaluation

| Metric | Value |
|---|---:|
| MAE | {evaluation.mae:.2f} K |
| RMSE | {evaluation.rmse:.2f} K |
| R² | {evaluation.r2:.3f} |

Use the same test fraction and random seed when comparing models. These metrics describe only this split of the published dataset.
""")
        except Exception as exc:
            evaluation = None
            metrics = mo.md(f"**Evaluation failed:** {type(exc).__name__}: {exc}")
    else:
        evaluation, metrics = None, mo.md("Load the dataset to evaluate a model.")
    metrics
    return (evaluation,)

@app.cell
def _(evaluation, plt):
    if evaluation is not None:
        fig, ax = plt.subplots()
        ax.scatter(evaluation.y_test, evaluation.y_pred, alpha=0.65, s=22)
        lower = min(float(evaluation.y_test.min()), float(evaluation.y_pred.min()))
        upper = max(float(evaluation.y_test.max()), float(evaluation.y_pred.max()))
        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_xlabel("Measured crystallization onset Tx (K)")
        ax.set_ylabel("Predicted Tx (K)")
        ax.set_title(f"Measured vs predicted — {evaluation.model_name}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        display_plot = fig
    else:
        display_plot = "The plot appears after successful evaluation."
    display_plot
    return

@app.cell
def _(evaluation, mo, np):
    if evaluation is not None:
        absolute_error = np.abs(evaluation.y_pred - evaluation.y_test)
        summary = mo.md(f"""## Error analysis

- Median absolute error: **{float(np.median(absolute_error)):.2f} K**
- 90th percentile absolute error: **{float(np.quantile(absolute_error, 0.9)):.2f} K**
- Test observations: **{len(evaluation.y_test)}**

A single random split may overestimate performance on genuinely new alloy families. Composition alone omits factors such as heating rate and processing history. Errors are not uncertainty intervals.
""")
    else:
        summary = mo.md("Error analysis appears after successful evaluation.")
    summary
    return

@app.cell
def _(X, mo, y):
    if X is not None and y is not None:
        formula = mo.ui.text(value="Fe72B20Si8", label="Alloy formula (atomic amounts, e.g. Fe72B20Si8)")
        predict_button = mo.ui.run_button(label="Estimate crystallization onset Tx")
        mo.vstack([formula, predict_button])
    else:
        formula, predict_button = None, None
    return formula, predict_button

@app.cell
def _(X, formula, mo, model_choice, predict_button, predict_tx, seed, y):
    if X is not None and formula is not None and predict_button.value and model_choice is not None:
        try:
            estimate, elements = predict_tx(X, y, model_choice.value, formula.value, random_state=int(seed.value))
            result = mo.md(f"""## Exploratory estimate

**Estimated Tx: {estimate:.1f} K**

Parsed elements: {', '.join(elements)}.

This estimate is generated from the selected model trained on all labeled rows. It is not a calibrated confidence interval. Check that the composition is represented by the training dataset and validate experimentally before relying on it.
""")
        except Exception as exc:
            result = mo.md(f"**Could not estimate Tx:** {type(exc).__name__}: {exc}")
    else:
        result = mo.md("Enter a composition and click **Estimate crystallization onset Tx** after the dataset loads.")
    result
    return

if __name__ == "__main__":
    app.run()
