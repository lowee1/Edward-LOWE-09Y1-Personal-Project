import requests
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import Dense
import datetime

def getDataDownloaded():
    df = pd.read_csv("C:\\Users\\Student\\Downloads\\feeds.csv")
    df = df.dropna(how='any',axis=0) 
    df.reset_index()

    y = df[["field1","field2","field3","field4"]]
    y = y.drop(["field2"],axis=1)
    datetimeString = pd.DataFrame(df["created_at"].str.split(" ").tolist(),columns=["date","time","timezone"])
    datetimeString = datetimeString.drop(columns=["timezone"])
    xDate = pd.DataFrame(datetimeString.date.str.split("-").tolist(),columns=["year","month","day"]).astype(int)
    xTime = pd.DataFrame(datetimeString.time.str.split(":").tolist(),columns=["hour","minute","second"]).astype(int)
    x = pd.concat([xDate,xTime], axis=1)
    return x,y


def getDataWeb(timeOfDay):
    response = requests.get("http://api.thingspeak.com/channels/472950/feeds.json?api_key=3PSUUNMTCVE6FVJO&days=0&results=8000&minutes=100&status=false&metadata=false&timescale=daily&start=2018-05-01%20"+timeOfDay)
    df = pd.DataFrame.from_dict(response.json()["feeds"])
    df = df.dropna(how='any',axis=0) 
    df.reset_index()

    y = df[["field1","field3","field4"]]

    datetimeString = pd.DataFrame(df["created_at"].str.split("T").tolist(),columns=["date","time"])
    datetimeString["time"] = datetimeString["time"].str[:-1]
    xDate = pd.DataFrame(datetimeString.date.str.split("-").tolist(),columns=["year","month","day"]).astype(int)
    xTime = pd.DataFrame(datetimeString.time.str.split(":").tolist(),columns=["hour","minute","second"]).astype(int)
    x = pd.concat([xDate,xTime], axis=1)
    return x,y

def makeModel(x,y):
    global source
    sequentialModel = Sequential()
    sequentialModel.add(Dense(800,input_dim=6,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(216,activation="relu"))
    sequentialModel.add(Dense(3,activation="linear"))

    sequentialModel.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])

    fit_epochs = {"downloaded":10,"web":5}
    sequentialModel.fit(x,y,epochs=fit_epochs[source])

    return sequentialModel

def saveModel():
    model_json = model.to_json()
    with open("weathermodel.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("weathermodel.h5")
    print("Saved model to disk")

def loadModel():
    with open("weathermodel.json","r") as file:
        model_json = file.read()
        loaded_model = model_from_json(model_json)
    loaded_model.load_weights("weathermodel.h5")

    return loaded_model


if input("Where should I get the model from (make or load): ") == "load":
    model = loadModel()
    source = "load"
elif input("Should I get the data from web or the downloaded: ") == "downloaded":
    x,y = getDataDownloaded()
    source = "downloaded"
    model = makeModel(x,y)
else:
    source = "web"

time = 0
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

    if source == "web" and time == 0:
        model = makeModel(x,y)
        model = makeModel(getDataWeb(str(hour)+":"+str(minute)+":"+str(second)))
        time = 1000

    userQuery = pd.DataFrame({"year":[year],"month":[month],"day":[day],"hour":[hour],"minute":[minute],"second":[second]})

    prediction = model.predict(userQuery)

    print("My prediction for",year,month,day,hour,minute,second,"is",prediction)

print("The End")