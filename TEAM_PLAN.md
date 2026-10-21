# Two-person team plan

Deadline: 22 October 2026, 7:00 PM CDT (23 October, 8:00 AM Singapore time).

## Teammate A — Data and modelling
- Verify the UCI dataset loads.
- Review data cleaning and target selection.
- Run Median Baseline, Ridge Regression, and Extra Trees.
- Record MAE, RMSE, R², test fraction, random seed, and sample counts.
- Note limitations; do not claim causality or experimental discovery.

## Teammate B — Interface and submission
- Launch the Marimo app and check each UI control.
- Review charts and error analysis.
- Test setup from a fresh environment and improve README if needed.
- Prepare a demo under four minutes, supporting screenshots, social post, and one-page write-up.

## Integration protocol
- Work on separate branches or separate files where possible.
- Commit small, meaningful changes.
- Open a pull request before merging.
- Run `python -m unittest discover -s tests -v` after integration.
- Both teammates should review metrics and write-up before submission.

## Suggested order
1. Confirm package installation and unit tests.
2. Run app and fetch real data.
3. Validate model metrics.
4. Finish README and public repository.
5. Capture the demo and write-up.
6. Publish social post and submit.
