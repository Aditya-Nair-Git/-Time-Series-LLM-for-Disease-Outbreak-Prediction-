import argparse
from pathlib import Path
import pandas as pd
from src.models.train import train_models
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input",default="data/processed/cleaned_disease_data.csv"); p.add_argument("--output",default="artifacts"); p.add_argument("--regressor",choices=["random_forest","xgboost"],default="random_forest"); a=p.parse_args()
 if not Path(a.input).exists(): raise SystemExit(f"Dataset not found: {a.input}. Run scripts/prepare_data.py first.")
 m=train_models(pd.read_csv(a.input),a.output,a.regressor); print(f"Artifacts saved. Accuracy={m['classification']['accuracy']:.3f}; MAE={m['regression']['mae']:.3f}")
if __name__=="__main__": main()
