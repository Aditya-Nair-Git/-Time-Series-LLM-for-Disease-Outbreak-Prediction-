# Report appendix implementation notes

This index is based on Chapter 8 of the uploaded capstone report, printed pages 51–96. It records what is actually shown in the report and how it maps to the reconstructed project. It is not a verbatim replacement for the full PDF.

## Code covered by the report

- **Pages 51–55 — preprocessing and feature engineering:** numeric conversion and imputation for `Cases`, `Deaths`, `preci`, `LAI`, and `Temp`; week extraction from `week_of_outbreak`; disease-label mapping; prompt/target preparation for LLM training; lag and rolling features; time-series cross-validation.
- **Pages 56–63 — ML and deep learning:** XGBoost regressor configuration and an `OptimizedGRUModel` with GRU layers, batch normalisation, dense layers, dropout, early stopping and evaluation/plotting code.
- **Pages 64–84 — LLM experiments:** prompt formatting, training/validation and prediction examples for the LLM-based numerical forecasting experiments.
- **Pages 85–88 — FastAPI backend:** model artifact loading, Pydantic request schemas, input preparation, `/predict_disease`, `/predict_cases` and health-check routes.
- **Pages 88–91 — React dashboard snippets:** calls the API, stores disease and case forecasts, shows dashboard cards/charts and an India map with at-risk districts.
- **Pages 91–96 — final classifier/regressor code:** week extraction, temperature conversion, feature selection, train/test split, Random Forest classification and regression, evaluation, and model serialization.

## Representative code patterns shown in the report

### Week parsing

The appendix extracts the week number from strings such as “1st week” or “52nd week” using a regular expression:

```python
df["week_of_year"] = df["week_of_outbreak"].str.extract(r"(\\d+)").astype(int)
```

### Temperature conversion

The report's final-classifier section describes converting Kelvin to Celsius:

```python
df["temp_celsius"] = df["Temp"] - 273.15
```

The reconstruction handles likely Kelvin values while leaving values already in Celsius unchanged, because the report also describes mixed temperature formats.

### Random Forest classifier and artifact saving

The appendix uses a train/test split and a Random Forest classifier, evaluates accuracy/classification metrics, and saves the estimator and its expected columns with joblib. The rebuilt pipeline keeps the preprocessing steps inside the saved scikit-learn pipeline to reduce inference-time column mismatch.

### API contract

The report specifically describes `/predict_disease` for the disease label and `/predict_cases` for the expected weekly count. The reconstructed API keeps these endpoint names and adds `/forecast` for a multi-week series used by the dashboard.

## Important reproducibility note

The appendix contains snippets from multiple experimental iterations, including differing model configurations and LLM experiments. The report does not include the original `Final_data.csv`, all training environment details, or the fitted model artifacts in the repository snapshot. Therefore this project implements a consistent runnable path from the documented approach; it does not claim to be a byte-for-byte restoration of every experiment or to reproduce every reported score.
