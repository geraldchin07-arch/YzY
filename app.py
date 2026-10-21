"""Marimo interface for MatSci Lens."""
import marimo
app = marimo.App(width="medium")

@app.cell
def _():
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    from src.matsci_lens import evaluate_model, load_uci_superconductivity
    return evaluate_model, load_uci_superconductivity, mo, np, plt

@app.cell
def _(mo):
    mo.md("""# MatSci Lens\n### Exploring machine-learning predictions of superconducting critical temperature\n\nCompare a simple baseline with machine-learning models using held-out data. This is a research exploration tool—not experimental confirmation of a material.""")
    return

@app.cell
def _(mo):
    load_button = mo.ui.run_button(label="Load UCI dataset")
    load_button
    return (load_button,)

@app.cell
def _(load_button, load_uci_superconductivity, mo):
    if load_button.value:
        try:
            X, y = load_uci_superconductivity()
            load_message = mo.md(f"**Dataset loaded.** {len(X):,} rows · {X.shape[1]} numeric feature columns · Target median: {y.median():.3f} K.")
        except Exception as exc:
            X, y = None, None
            load_message = mo.md(f"**Could not load dataset.** {type(exc).__name__}: {exc}\n\nCheck internet access and try again.")
    else:
        X, y = None, None
        load_message = mo.md("Click **Load UCI dataset** to download and prepare the data.")
    load_message
    return X, y

@app.cell
def _(X, mo, y):
    if X is not None and y is not None:
        model_choice = mo.ui.dropdown(options=["Median Baseline", "Ridge Regression", "Extra Trees"], value="Extra Trees", label="Model")
        test_fraction = mo.ui.slider(start=0.1, stop=0.3, step=0.05, value=0.2, label="Test-set fraction")
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
            metrics = mo.md(f"""## Held-out test performance\n\n| Metric | Value |\n|---|---:|\n| MAE | {evaluation.mae:.3f} K |\n| RMSE | {evaluation.rmse:.3f} K |\n| R² | {evaluation.r2:.3f} |\n\nRun all model choices with the same test fraction and seed before comparing them.""")
        except Exception as exc:
            evaluation = None
            metrics = mo.md(f"**Evaluation failed.** {type(exc).__name__}: {exc}")
    else:
        evaluation, metrics = None, mo.md("Load the dataset to run an evaluation.")
    metrics
    return (evaluation,)

@app.cell
def _(evaluation, plt):
    if evaluation is not None:
        fig, ax = plt.subplots()
        ax.scatter(evaluation.y_test, evaluation.y_pred, alpha=0.35, s=12)
        lower = min(float(evaluation.y_test.min()), float(evaluation.y_pred.min()))
        upper = max(float(evaluation.y_test.max()), float(evaluation.y_pred.max()))
        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_xlabel("Measured critical temperature (K)")
        ax.set_ylabel("Predicted critical temperature (K)")
        ax.set_title(f"Measured vs predicted — {evaluation.model_name}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        display_plot = fig
    else:
        display_plot = "The plot will appear after a successful evaluation."
    display_plot
    return

@app.cell
def _(evaluation, mo, np):
    if evaluation is not None:
        absolute_error = np.abs(evaluation.y_pred - evaluation.y_test)
        summary = mo.md(f"""## Error analysis\n\n- Median absolute error: **{float(np.median(absolute_error)):.3f} K**\n- 90th percentile absolute error: **{float(np.quantile(absolute_error, 0.9)):.3f} K**\n- Test observations: **{len(evaluation.y_test):,}**\n\nThese errors describe this held-out test set; they do not establish reliability for every new material. Feature association is not causation, and experimental validation remains necessary.""")
    else:
        summary = mo.md("Error analysis will appear after a successful evaluation.")
    summary
    return

if __name__ == "__main__":
    app.run()
