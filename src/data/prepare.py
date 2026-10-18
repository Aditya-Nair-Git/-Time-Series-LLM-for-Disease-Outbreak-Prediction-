"""Prepare epidemiology CSV into a consistent schema."""
import re
import pandas as pd
REQUIRED={"Disease","Cases","week_of_outbreak","state_ut","district","Temp","preci"}
def parse_week(value):
    if pd.isna(value): return None
    m=re.search(r"\d+",str(value)); return int(m.group()) if m else None
def prepare_frame(df,exclude_pandemic_years=True):
    df=df.copy(); missing=REQUIRED-set(df.columns)
    if missing: raise ValueError("Missing required columns: "+", ".join(sorted(missing)))
    for col in ["Cases","Deaths","Temp","preci","LAI","Latitude","Longitude","year","month"]:
        if col in df: df[col]=pd.to_numeric(df[col].astype(str).str.replace(",","",regex=False),errors="coerce")
    df["week_of_year"]=df["week_of_outbreak"].map(parse_week)
    # Convert values that look like Kelvin; already-Celsius values are left unchanged.
    df["temp_celsius"]=df["Temp"].where(df["Temp"]<=100,df["Temp"]-273.15)
    if exclude_pandemic_years and "year" in df: df=df.loc[~df["year"].isin([2020,2021,2022])].copy()
    df["lai"]=df["LAI"] if "LAI" in df else 1.0
    df["latitude"]=df["Latitude"] if "Latitude" in df else 20.0
    df["longitude"]=df["Longitude"] if "Longitude" in df else 80.0
    for col in ["Disease","state_ut","district"]: df[col]=df[col].astype("string").str.strip()
    cols=["Disease","Cases","week_of_year","state_ut","district","temp_celsius","preci","lai","latitude","longitude"]
    if "year" in df: cols.append("year")
    out=df[cols].dropna(subset=["Disease","Cases","week_of_year","state_ut","district","temp_celsius","preci"])
    out=out.loc[out["week_of_year"].between(1,53)&out["Cases"].ge(0)].drop_duplicates().copy()
    out["week_of_year"]=out["week_of_year"].astype(int)
    if out.empty: raise ValueError("No usable rows remain; check input schema.")
    return out.reset_index(drop=True)
def load_and_prepare(path,exclude_pandemic_years=True):
    return prepare_frame(pd.read_csv(path),exclude_pandemic_years)
