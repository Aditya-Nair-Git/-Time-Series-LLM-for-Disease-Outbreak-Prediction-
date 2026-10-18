import pandas as pd
from src.data.prepare import prepare_frame
def test_cleaning_and_temperature_conversion():
 df=pd.DataFrame({"Disease":["Dengue","Malaria"],"Cases":[5,8],"week_of_outbreak":["40th week","12"],"state_ut":["Andhra Pradesh","Kerala"],"district":["Guntur","Ernakulam"],"Temp":[301.65,28.0],"preci":[50.,20.],"year":[2019,2019]})
 out=prepare_frame(df); assert out["week_of_year"].tolist()==[40,12]; assert round(out.iloc[0]["temp_celsius"],1)==28.5; assert round(out.iloc[1]["temp_celsius"],1)==28.0
