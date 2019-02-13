import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import time
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

engine = create_engine('mysql+pymysql://root:guangzhou@192.168.55.250:3306/fzwdata')


sql = '''
SELECT pool_id,datatimestr,sum(sgwusers),sum(pgwusers)
FROM `saegw_users`,`saegw_name`
where DATEDIFF(datatimestr,NOW())<=0 AND DATEDIFF(datatimestr,NOW())>-6 and ggsnname = database_name and pool_id in (1,2,3,4)
GROUP BY pool_id,datatimestr
ORDER BY datatimestr
      '''

# read_sql_query的两个参数: sql语句， 数据库连接
dataset = pd.read_sql_query(sql, engine)


# 1 guangzhou pool; 2 yuexi pool ; 3 shenzhen pool;  4 yuedong pool;
pool1 = dataset[dataset.pool_id==1]
pool2 = dataset[dataset.pool_id==2]
pool3 = dataset[dataset.pool_id==3]
pool4 = dataset[dataset.pool_id==4]


# X is timestamp
X1 = pool1.iloc[:,1:2].values.astype(np.int64)/ 10 ** 9
X2 = pool2.iloc[:,1:2].values.astype(np.int64)/ 10 ** 9
X3 = pool3.iloc[:,1:2].values.astype(np.int64)/ 10 ** 9
X4 = pool4.iloc[:,1:2].values.astype(np.int64)/ 10 ** 9

pool_id_x1 = pool1.iloc[:,0:1].values
pool_id_x2 = pool2.iloc[:,0:1].values
pool_id_x3 = pool3.iloc[:,0:1].values
pool_id_x4 = pool4.iloc[:,0:1].values

sgwy1 = pool1.iloc[:,2].values
sgwy2 = pool2.iloc[:,2].values
sgwy3 = pool3.iloc[:,2].values
sgwy4 = pool4.iloc[:,2].values

pgwy1 = pool1.iloc[:,3].values
pgwy2 = pool2.iloc[:,3].values
pgwy3 = pool3.iloc[:,3].values
pgwy4 = pool4.iloc[:,3].values




poly_reg_sgw1 = PolynomialFeatures(degree = 4)
poly_reg_sgw2 = PolynomialFeatures(degree = 4)
poly_reg_sgw3 = PolynomialFeatures(degree = 4)
poly_reg_sgw4 = PolynomialFeatures(degree = 4)

poly_reg_pgw1 = PolynomialFeatures(degree = 4)
poly_reg_pgw2 = PolynomialFeatures(degree = 4)
poly_reg_pgw3 = PolynomialFeatures(degree = 4)
poly_reg_pgw4 = PolynomialFeatures(degree = 4)

#sgw
X_poly_sgw1 = poly_reg_sgw1.fit_transform(X1)
poly_reg_sgw1.fit(X_poly_sgw1, sgwy1)
lin_reg_sgw1 = LinearRegression()
lin_reg_sgw1.fit(X_poly_sgw1, sgwy1)

X_poly_sgw2 = poly_reg_sgw2.fit_transform(X2)
poly_reg_sgw2.fit(X_poly_sgw2, sgwy2)
lin_reg_sgw2 = LinearRegression()
lin_reg_sgw2.fit(X_poly_sgw2, sgwy2)

X_poly_sgw3 = poly_reg_sgw3.fit_transform(X3)
poly_reg_sgw3.fit(X_poly_sgw3, sgwy3)
lin_reg_sgw3 = LinearRegression()
lin_reg_sgw3.fit(X_poly_sgw3, sgwy3)

X_poly_sgw4 = poly_reg_sgw4.fit_transform(X4)
poly_reg_sgw4.fit(X_poly_sgw4, sgwy4)
lin_reg_sgw4 = LinearRegression()
lin_reg_sgw4.fit(X_poly_sgw4, sgwy4)

#pgw
X_poly_pgw1 = poly_reg_pgw1.fit_transform(X1)
poly_reg_pgw1.fit(X_poly_pgw1, pgwy1)
lin_reg_pgw1 = LinearRegression()
lin_reg_pgw1.fit(X_poly_pgw1, pgwy1)

X_poly_pgw2 = poly_reg_pgw2.fit_transform(X2)
poly_reg_pgw2.fit(X_poly_pgw2, pgwy2)
lin_reg_pgw2 = LinearRegression()
lin_reg_pgw2.fit(X_poly_pgw2, pgwy2)

X_poly_pgw3 = poly_reg_pgw3.fit_transform(X3)
poly_reg_pgw3.fit(X_poly_pgw3, pgwy3)
lin_reg_pgw3 = LinearRegression()
lin_reg_pgw3.fit(X_poly_pgw3, pgwy3)

X_poly_pgw4 = poly_reg_pgw4.fit_transform(X4)
poly_reg_pgw4.fit(X_poly_pgw4, pgwy4)
lin_reg_pgw4 = LinearRegression()
lin_reg_pgw4.fit(X_poly_pgw4, pgwy4)


#use 3days timestamp to pridict usernumbers
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
yesterday_end_time = int(time.mktime(time.strptime(str(today), '%Y-%m-%d'))) - 1
today_start_time = yesterday_end_time + 1

daysToCome = []

#863 = 3days * 60min / 5per
for i in range(0,864):
    daysToCome.append(today_start_time+i*300)

#print(daysToCome)

df = pd.DataFrame({'future3ds':daysToCome})

df1 = pd.DataFrame({'future3ds':daysToCome})
df2 = pd.DataFrame({'future3ds':daysToCome})
df3 = pd.DataFrame({'future3ds':daysToCome})
df4 = pd.DataFrame({'future3ds':daysToCome})

df1['future3ds'] = 1;
df2['future3ds'] = 2;
df3['future3ds'] = 3;
df4['future3ds'] = 4;

print(df1)
#print(type(df))

#change to matrix
t = df.values
t1 = df1.values #all 1
t2 = df2.values  # all 2
t3 = df3.values # all 3
t4 = df4.values # all 4

#t + X1
all_time1 = np.vstack((X1,t))
all_time2 = np.vstack((X2,t))
all_time3 = np.vstack((X3,t))
all_time4 = np.vstack((X4,t))

all_pool_id1 = np.vstack((pool_id_x1,t1))
all_pool_id2 = np.vstack((pool_id_x2,t2))
all_pool_id3 = np.vstack((pool_id_x3,t3))
all_pool_id4 = np.vstack((pool_id_x4,t4))





#generate predicted
predictSgw1Values = lin_reg_sgw1.predict(poly_reg_sgw1.fit_transform(all_time1))
predictPgw1Values = lin_reg_pgw1.predict(poly_reg_pgw1.fit_transform(all_time1))

predictSgw2Values = lin_reg_sgw2.predict(poly_reg_sgw2.fit_transform(all_time2))
predictPgw2Values = lin_reg_pgw2.predict(poly_reg_pgw2.fit_transform(all_time2))

predictSgw3Values = lin_reg_sgw3.predict(poly_reg_sgw3.fit_transform(all_time3))
predictPgw3Values = lin_reg_pgw3.predict(poly_reg_pgw3.fit_transform(all_time3))

predictSgw4Values = lin_reg_sgw4.predict(poly_reg_sgw4.fit_transform(all_time4))
predictPgw4Values = lin_reg_pgw4.predict(poly_reg_pgw4.fit_transform(all_time4))



#test = np.array([datetime.utcfromtimestamp(t)  for t in all_time1])
#testFormatTime = all_time1.astype(np.datetime64)

stampTimeItemAll1=[]
stampTimeItemAll2=[]
stampTimeItemAll3=[]
stampTimeItemAll4=[]

for stampTimeItem in all_time1:
    stampTimeItemAll1.append(datetime.datetime.utcfromtimestamp(stampTimeItem))

for stampTimeItem in all_time2:
    stampTimeItemAll2.append(datetime.datetime.utcfromtimestamp(stampTimeItem))

for stampTimeItem in all_time3:
    stampTimeItemAll3.append(datetime.datetime.utcfromtimestamp(stampTimeItem))

for stampTimeItem in all_time4:
    stampTimeItemAll4.append(datetime.datetime.utcfromtimestamp(stampTimeItem))

#print(stampTimeItemAll)


# print(len(all_time1.tolist()))
# print(len(all_pool_id1.T.tolist()))
# print(len(predictSgw1Values.tolist()))
# print(len(predictPgw1Values.tolist()))

#save data to db

df_save = pd.DataFrame({'datatime_predict':stampTimeItemAll1,'pool_id':all_pool_id1.tolist(),'sgw_users_predict':predictSgw1Values.tolist(),'pgw_users_predict':predictPgw1Values.tolist()})
df_save.to_sql('pool_users_predict', engine,  if_exists = 'replace', index= False)


df_save = pd.DataFrame({'datatime_predict':stampTimeItemAll2,'pool_id':all_pool_id2.tolist(),'sgw_users_predict':predictSgw2Values.tolist(),'pgw_users_predict':predictPgw2Values.tolist()})
df_save.to_sql('pool_users_predict', engine,  if_exists = 'replace', index= False)

df_save = pd.DataFrame({'datatime_predict':stampTimeItemAll3,'pool_id':all_pool_id3.tolist(),'sgw_users_predict':predictSgw3Values.tolist(),'pgw_users_predict':predictPgw3Values.tolist()})
df_save.to_sql('pool_users_predict', engine,  if_exists = 'replace', index= False)

df_save = pd.DataFrame({'datatime_predict':stampTimeItemAll4,'pool_id':all_pool_id4.tolist(),'sgw_users_predict':predictSgw4Values.tolist(),'pgw_users_predict':predictPgw4Values.tolist()})
df_save.to_sql('pool_users_predict', engine,  if_exists = 'replace', index= False)


# Visualising the Polynomial Regression results
# plt.figure()
# plt.scatter(X1, sgwy1, color = 'green')
# plt.plot(X1, lin_reg_sgw1.predict(poly_reg_sgw1.fit_transform(X1)), color = 'blue')
# plt.plot(all_time1, lin_reg_sgw1.predict(poly_reg_sgw1.fit_transform(all_time1)), color = 'red')
# plt.title('users GuangZhou pool(Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('sgwusers')


# plt.scatter(X1, pgwy1, color = 'greenyellow')
# plt.plot(X1, lin_reg_pgw1.predict(poly_reg_pgw1.fit_transform(X1)), color = 'blue')
# plt.plot(t, lin_reg_pgw1.predict(poly_reg_pgw1.fit_transform(t)), color = 'red')

# plt.show()


# plt.figure()
# plt.scatter(X2, sgwy2, color = 'green')
# plt.plot(X2, lin_reg_sgw2.predict(poly_reg_sgw2.fit_transform(X1)), color = 'blue')
# plt.plot(t, lin_reg_sgw2.predict(poly_reg_sgw2.fit_transform(t)), color = 'red')
# plt.title('users YueXi pool(Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('sgwusers')


# plt.scatter(X1, pgwy2, color = 'greenyellow')
# plt.plot(X1, lin_reg_pgw2.predict(poly_reg_pgw2.fit_transform(X1)), color = 'blue')
# plt.plot(t, lin_reg_pgw2.predict(poly_reg_pgw2.fit_transform(t)), color = 'red')

# plt.show()



# plt.figure()
# plt.scatter(X4, sgwy4, color = 'green')
# plt.plot(X4, lin_reg_sgw4.predict(poly_reg_sgw4.fit_transform(X4)), color = 'blue')
# plt.plot(t, lin_reg_sgw4.predict(poly_reg_sgw4.fit_transform(t)), color = 'red')
# plt.title('users DongGuan pool(Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('sgwusers')


# plt.scatter(X4, pgwy4, color = 'greenyellow')
# plt.plot(X4, lin_reg_pgw4.predict(poly_reg_pgw4.fit_transform(X4)), color = 'blue')
# plt.plot(t, lin_reg_pgw4.predict(poly_reg_pgw4.fit_transform(t)), color = 'red')

# plt.show()


# plt.figure()
# plt.scatter(X3, sgwy3, color = 'green')
# plt.plot(X3, lin_reg_sgw3.predict(poly_reg_sgw3.fit_transform(X3)), color = 'blue')
# plt.plot(t, lin_reg_sgw3.predict(poly_reg_sgw3.fit_transform(t)), color = 'red')
# plt.title('users ShenZhen pool(Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('sgwusers')


# plt.scatter(X3, pgwy3, color = 'greenyellow')
# plt.plot(X3, lin_reg_pgw3.predict(poly_reg_pgw3.fit_transform(X3)), color = 'blue')
# plt.plot(t, lin_reg_pgw3.predict(poly_reg_pgw3.fit_transform(t)), color = 'red')

# plt.show()

