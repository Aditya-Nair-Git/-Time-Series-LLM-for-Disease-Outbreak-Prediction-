"""Train Random Forest classification and a configurable case-count regressor."""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,mean_absolute_error,mean_squared_error,r2_score
BASE=["week_of_year","state_ut","district","temp_celsius","preci","lai","latitude","longitude"]
CAT=["state_ut","district"]; NUM=[c for c in BASE if c not in CAT]
def processor(cat,num):
    return ColumnTransformer([("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat),("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num)])
def train_models(df,output_dir="artifacts",regressor="random_forest"):
    missing=set(BASE+["Disease","Cases"])-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); metrics={"rows":len(df)}
    clf=Pipeline([("prep",processor(CAT,NUM)),("model",RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1))])
    x=df[BASE]; y=df["Disease"].astype(str); strat=y if y.value_counts().min()>=2 else None
    a,b,c,d=train_test_split(x,y,test_size=.2,random_state=42,stratify=strat); clf.fit(a,c); yp=clf.predict(b)
    metrics["classification"]={"accuracy":float(accuracy_score(d,yp)),"report":classification_report(d,yp,output_dict=True,zero_division=0),"split":"random 80/20 baseline"}
    joblib.dump(clf,out/"disease_model.joblib")
    x=df[BASE+["Disease"]]; y=df["Cases"].astype(float); cat=CAT+["Disease"]
    if regressor=="xgboost":
        try: from xgboost import XGBRegressor
        except ImportError as e: raise RuntimeError("Install optional XGBoost with pip install xgboost") from e
        model=XGBRegressor(n_estimators=500,max_depth=5,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective="reg:squarederror",random_state=42,n_jobs=-1)
    else: model=RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=-1)
    reg=Pipeline([("prep",processor(cat,NUM)),("model",model)])
    a,b,c,d=train_test_split(x,y,test_size=.2,random_state=42); reg.fit(a,c); pred=np.maximum(0,reg.predict(b))
    metrics["regression"]={"mae":float(mean_absolute_error(d,pred)),"rmse":float(np.sqrt(mean_squared_error(d,pred))),"r2":float(r2_score(d,pred)),"model":regressor,"split":"random 80/20 baseline"}
    joblib.dump(reg,out/"case_model.joblib")
    meta={"rows":len(df),"classifier":"RandomForestClassifier","regressor":regressor,"metrics":{"classification_accuracy":metrics["classification"]["accuracy"],"regression_mae":metrics["regression"]["mae"],"regression_rmse":metrics["regression"]["rmse"],"regression_r2":metrics["regression"]["r2"]},"warning":"Random split is not future-time validation"}
    joblib.dump(meta,out/"model_metadata.joblib"); (out/"metrics.json").write_text(json.dumps(metrics,indent=2,default=float),encoding="utf-8")
    return metrics
