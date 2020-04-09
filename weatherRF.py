import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime

def getDataAndMakeModel(timeOfDay):
    response = requests.get("http://api.thingspeak.com/channels/472950/feeds.json?results=8000&days=0&minutes=100&status=false&metadata=false&timescale=daily&start=2018-05-01%20"+timeOfDay)
    df = pd.DataFrame.from_dict(response.json()["feeds"])
    df = df.dropna(how='any',axis=0) 
    df.reset_index()

    y = df[["field1","field2","field3","field4"]]

    datetimeString = pd.DataFrame(df["created_at"].str.split("T").tolist(),columns=["date","time"])
    datetimeString["time"] = datetimeString["time"].str[:-1]
    xDate = pd.DataFrame(datetimeString.date.str.split("-").tolist(),columns=["year","month","day"]).astype(int)
    xTime = pd.DataFrame(datetimeString.time.str.split(":").tolist(),columns=["hour","minute","second"]).astype(int)
    x = pd.concat([xDate,xTime], axis=1)

    rf_model = RandomForestRegressor()
    rf_model.fit(x,y)
    return rf_model

while True:
    try:
        year = input("Year: ")
        if year == "stop":
            break
        else:
            year = int(year)
        month = int(input("Month: "))
        day = int(input("Day: "))
        hour = int(input("Hour (zero-padded): "))
        minute = int(input("Minute: "))
        second = int(input("Second: "))
    except:
        continue
    
    rf_model = getDataAndMakeModel(str(hour)+":"+str(minute)+":"+str(second))

    userQuery = pd.DataFrame({"year":[year],"month":[month],"day":[day],"hour":[hour],"minute":[minute],"second":[second]})

    prediction = rf_model.predict(userQuery)

    print("My prediction for",year,month,day,hour,minute,second,"is",prediction)

print("The End")