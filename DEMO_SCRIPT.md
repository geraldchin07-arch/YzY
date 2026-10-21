# CrystalCast demo script (2–3 minutes)

1. **Problem (20 s):** “We explore whether the composition of an iron-based metallic-glass alloy can help predict the temperature at which crystallization begins.”
2. **Dataset (20 s):** Load the open RWTH dataset. Show the valid measurement count and composition-feature count.
3. **Model evaluation (45 s):** Use the same test fraction and random seed to evaluate the Median Baseline, Ridge Regression, Extra Trees, and Transformer Neural Network. Read only the actual scores shown by the app.
4. **Prediction plot (25 s):** Explain that points near the diagonal have more accurate predictions; points farther away are larger errors.
5. **User-input demo (30 s):** Enter a composition in the simple element-plus-amount format, such as Fe72B20Si8, only if those elements are present in the loaded data. Explain that the result is an exploratory estimate.
6. **Limitations (20 s):** The Transformer is experimental and may not outperform simpler models. The model uses composition only, cannot replace experiments, and may generalize poorly to new alloy families.

Record only after confirming the live dataset and all UI controls work. Keep the submitted video at or below four minutes.
