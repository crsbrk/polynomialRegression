import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import time
import datetime

ts = time.time()
print(ts)
timeNow = datetime.datetime.now()
print(timeNow)

today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
print(tomorrow)
tomorrow_start_time = int(time.mktime(time.strptime(str(tomorrow), '%Y-%m-%d')))
print(tomorrow_start_time)

yesterday_end_time = int(time.mktime(time.strptime(str(today), '%Y-%m-%d'))) - 1
today_start_time = yesterday_end_time + 1

print(today_start_time)
daysToCome = []
#863 = 3days * 60min / 5per
for i in range(0,864):
    print (i)
    daysToCome.append(today_start_time+i*300)

print(daysToCome)

t = [[1548219900]]
print(t)

df = pd.DataFrame({'future3ds':daysToCome})
print(df)
print(type(df))

tMatrix = df.as_matrix()
print(tMatrix)