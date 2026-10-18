import argparse
from pathlib import Path
from src.data.prepare import load_and_prepare
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",default="data/processed/cleaned_disease_data.csv"); p.add_argument("--keep-pandemic-years",action="store_true"); a=p.parse_args()
 df=load_and_prepare(a.input,not a.keep_pandemic_years); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); print(f"Saved {len(df):,} rows to {out}")
if __name__=="__main__": main()
