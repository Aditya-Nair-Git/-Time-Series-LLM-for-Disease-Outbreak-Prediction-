from fastapi.testclient import TestClient
from backend.main import app
client=TestClient(app)
p={"week_of_year":40,"state_ut":"Andhra Pradesh","district":"Guntur","temp_celsius":28.5,"preci":50.0,"lai":1.2,"latitude":16.3,"longitude":80.4}
def test_health(): assert client.get("/health").json()["ok"]
def test_forecast():
 r=client.post("/forecast?horizon=4",json=p); assert r.status_code==200 and len(r.json()["forecast"])==4
def test_cases():
 r=client.post("/predict_cases",json={**p,"Disease":"Dengue"}); assert r.status_code==200 and r.json()["predicted_cases"]>=0
