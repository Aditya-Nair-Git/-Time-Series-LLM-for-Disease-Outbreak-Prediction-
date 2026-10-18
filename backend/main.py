"""FastAPI outbreak prediction API; runs labeled demo mode until models are trained."""
from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/"artifacts"
logging.basicConfig(level=logging.INFO)
app=FastAPI(title="Time-Series LLM for Disease Outbreak Prediction",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:8501","http://127.0.0.1:8501"],allow_methods=["GET","POST"],allow_headers=["*"])

def load(name):
    try: return joblib.load(ARTIFACTS/name)
    except FileNotFoundError: return None
    except Exception: logging.exception("Could not load %s",name); return None

DISEASE_MODEL=load("disease_model.joblib")
CASE_MODEL=load("case_model.joblib")
METADATA=load("model_metadata.joblib") or {}

class DiseaseInput(BaseModel):
    week_of_year:int=Field(ge=1,le=53)
    state_ut:str=Field(min_length=1,max_length=100)
    district:str=Field(default="Unknown",min_length=1,max_length=100)
    temp_celsius:float=Field(ge=-20,le=60)
    preci:float=Field(ge=0,le=2000)
    lai:float=Field(default=1.0,ge=0,le=20)
    latitude:float=Field(default=20.0,ge=-90,le=90)
    longitude:float=Field(default=80.0,ge=-180,le=180)

class CaseInput(DiseaseInput):
    Disease:str=Field(min_length=1,max_length=100)

def features(x):
    return {"week_of_year":x.week_of_year,"state_ut":x.state_ut,"district":x.district,
            "temp_celsius":x.temp_celsius,"preci":x.preci,"lai":x.lai,
            "latitude":x.latitude,"longitude":x.longitude}

def demo_disease(x):
    if x.preci>=80 and x.temp_celsius>=24 and 20<=x.week_of_year<=40: return "Dengue",.52
    if x.temp_celsius>=27 and x.preci>=35: return "Acute Diarrhoeal Disease",.46
    if x.temp_celsius<18: return "Influenza-like illness",.41
    if x.preci>=45: return "Malaria",.39
    return "Other / monitor trend",.34

def get_disease(x):
    if DISEASE_MODEL is not None:
        label=str(DISEASE_MODEL.predict(pd.DataFrame([features(x)]))[0]); conf=None
        if hasattr(DISEASE_MODEL,"predict_proba"):
            try: conf=float(np.max(DISEASE_MODEL.predict_proba(pd.DataFrame([features(x)]))[0]))
            except Exception: pass
        return label,conf,"trained_model"
    label,conf=demo_disease(x)
    return label,conf,"demo"

def get_cases(x,disease=None):
    label=disease or x.Disease
    if CASE_MODEL is not None:
        row=features(x); row["Disease"]=label
        return max(0.0,float(CASE_MODEL.predict(pd.DataFrame([row]))[0])),"trained_model"
    factor={"Dengue":1.25,"Malaria":1.1,"Acute Diarrhoeal Disease":1.35,"Influenza-like illness":.85}.get(label,.75)
    season=1+.25*np.sin(2*np.pi*x.week_of_year/52)
    weather=max(.6,min(1.8,.7+x.preci/120+max(x.temp_celsius,0)/100))
    region=1+(sum(map(ord,x.state_ut))%7)/20
    return round(float(18*factor*season*weather*region),2),"demo"

@app.get("/")
def root():
    return {"name":app.title,"status":"ok","docs":"/docs","mode":"trained_model" if DISEASE_MODEL is not None and CASE_MODEL is not None else "demo"}

@app.get("/health")
def health():
    return {"ok":True,"disease_model_loaded":DISEASE_MODEL is not None,"case_model_loaded":CASE_MODEL is not None,
            "mode":"trained_model" if DISEASE_MODEL is not None and CASE_MODEL is not None else "demo","metadata":METADATA}

@app.post("/predict_disease")
def predict_disease(x:DiseaseInput):
    label,conf,mode=get_disease(x)
    return {"prediction":label,"confidence":conf,"mode":mode,
            "disclaimer":"Illustrative demo output, not validated epidemiology." if mode=="demo" else None}

@app.post("/predict_cases")
def predict_cases(x:CaseInput):
    estimate,mode=get_cases(x)
    return {"predicted_cases":round(estimate,2),"mode":mode,
            "disclaimer":"Illustrative demo output, not validated epidemiology." if mode=="demo" else None}

@app.post("/forecast")
def forecast(x:DiseaseInput,horizon:int=6):
    if not 1<=horizon<=26: raise HTTPException(status_code=422,detail="horizon must be between 1 and 26")
    disease,conf,dmode=get_disease(x); rows=[]; modes={dmode}
    for offset in range(horizon):
        values=features(x); values["week_of_year"]=(x.week_of_year-1+offset)%52+1; values["Disease"]=disease
        count,mode=get_cases(CaseInput(**values),disease); rows.append({"week_of_year":values["week_of_year"],"predicted_cases":round(count,2)}); modes.add(mode)
    run_mode="trained_model" if modes=={"trained_model"} else "demo"
    return {"disease":disease,"confidence":conf,"forecast":rows,"mode":run_mode,
            "disclaimer":"For demonstration only; not suitable for health decisions." if run_mode=="demo" else None}
