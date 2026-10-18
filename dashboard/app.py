import os, requests, pandas as pd, plotly.express as px, streamlit as st
st.set_page_config(page_title="Outbreak Forecast Studio",page_icon="🌦️",layout="wide")
api=os.getenv("OUTBREAK_API_URL","http://127.0.0.1:8000")
raw="https://raw.githubusercontent.com/Aditya-Nair-Git/-Time-Series-LLM-for-Disease-Outbreak-Prediction-/main/assets/"
st.markdown("""<div style="padding:26px;border-radius:18px;background:linear-gradient(120deg,#102a43,#176b87);color:white"><h1>🌦️ Outbreak Forecast Studio</h1><p>Climate-aware signals · Weekly estimates · Research prototype</p></div>""",unsafe_allow_html=True)
st.markdown("### Research overview")
col1,col2=st.columns(2)
with col1:
    st.markdown("**System architecture from the project report**")
    st.markdown(f'<img src="{raw}system-architecture.svg" style="width:100%;border-radius:12px" alt="System architecture diagram">',unsafe_allow_html=True)
with col2:
    st.markdown("**Reported regression comparison**")
    st.markdown(f'<img src="{raw}model-comparison.svg" style="width:100%;border-radius:12px" alt="Reported MAE comparison">',unsafe_allow_html=True)
st.caption("Comparison values are taken from Table 5.5.1 of the capstone report. They are not newly reproduced by this reconstruction.")
try: health=requests.get(api+"/health",timeout=2).json()
except Exception: health={"ok":False,"mode":"offline","disease_model_loaded":False,"case_model_loaded":False}
ready=health.get("disease_model_loaded") and health.get("case_model_loaded")
if not health.get("ok"): st.warning(f"API unavailable at {api}. Start it with uvicorn backend.main:app --reload")
elif not ready: st.info("Demo mode: outputs are illustrative, not validated outbreak forecasts.")
else: st.success("Trained model artifacts loaded.")
with st.sidebar:
    st.header("Scenario"); state=st.text_input("State / UT","Andhra Pradesh"); district=st.text_input("District","Guntur")
    week=st.slider("Week of year",1,53,40); temp=st.number_input("Temperature (°C)",-10.0,55.0,28.5)
    rain=st.number_input("Precipitation (mm)",0.0,1000.0,50.0); lai=st.number_input("Leaf Area Index",0.0,15.0,1.2)
    lat=st.number_input("Latitude",-90.0,90.0,16.3); lon=st.number_input("Longitude",-180.0,180.0,80.4)
    horizon=st.slider("Forecast horizon (weeks)",1,16,6); run=st.button("Generate forecast",type="primary",use_container_width=True)
payload={"week_of_year":week,"state_ut":state,"district":district,"temp_celsius":temp,"preci":rain,"lai":lai,"latitude":lat,"longitude":lon}
a,b,c,d=st.columns(4); a.metric("Region",f"{district}, {state}"); b.metric("Week",f"{week:02d}"); c.metric("Temperature",f"{temp:.1f} °C"); d.metric("Rainfall",f"{rain:.0f} mm")
if run:
    if not health.get("ok"): st.error("API is offline.")
    else:
        try:
            r=requests.post(api+"/forecast",params={"horizon":horizon},json=payload,timeout=30); r.raise_for_status(); st.session_state["latest"]=r.json()
        except requests.RequestException as e: st.error(f"Request failed: {e}")
result=st.session_state.get("latest")
if result:
    st.markdown("---"); st.subheader("Forecast results")
    c1,c2,c3=st.columns(3); c1.metric("Predicted disease",result["disease"]); conf=result.get("confidence"); c2.metric("Confidence",f"{conf:.0%}" if isinstance(conf,(int,float)) else "N/A"); c3.metric("Mode",result.get("mode","unknown").replace("_"," ").title())
    if result.get("disclaimer"): st.warning(result["disclaimer"])
    df=pd.DataFrame(result["forecast"]); x,y=st.columns(2)
    with x: st.plotly_chart(px.line(df,x="week_of_year",y="predicted_cases",markers=True,title="Weekly case-count estimate"),use_container_width=True)
    with y: st.plotly_chart(px.bar(df,x="week_of_year",y="predicted_cases",title="Week-by-week estimates"),use_container_width=True)
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("Download forecast CSV",df.to_csv(index=False).encode(),"outbreak_forecast.csv","text/csv")
else:
    st.markdown("---"); st.subheader("Explore a scenario")
    x,y,z=st.columns(3); x.markdown("#### 🦠 Disease signal\nEstimate a disease category from time and environmental inputs."); y.markdown("#### 📈 Weekly trend\nInspect case-count estimates across the forecast horizon."); z.markdown("#### 🧭 Regional context\nKeep state, district and coordinates attached to each scenario.")
with st.expander("Model/data status"): st.json(health)
with st.expander("Limitations"): st.write("Demo outputs use synthetic fallback rules. Trained models still require future-time validation and external evaluation. This prototype is not for diagnosis or real public-health decisions.")
