import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import time
import datetime

# 初始化数据库连接，使用pymysql模块
# MySQL的用户：root, 密码:147369, 端口：3306,数据库：mydb
engine = create_engine('mysql+pymysql://root:guangzhou@192.168.55.250:3306/fzwdata')

# 查询语句，选出employee表中的所有数据
sql = '''
SELECT pool_id,datatimestr,sum(sgwusers),sum(pgwusers)
FROM `saegw_users`,`saegw_name`
where DATEDIFF(datatimestr,NOW())<=0 AND DATEDIFF(datatimestr,NOW())>-6 and ggsnname = database_name and pool_id in (1,2,3,4)
GROUP BY pool_id,datatimestr
ORDER BY datatimestr
      '''

# read_sql_query的两个参数: sql语句， 数据库连接
dataset = pd.read_sql_query(sql, engine)

# 输出employee表的查询结果
#print(dataset)
# 1 guangzhou pool; 2 yuexi pool ; 3 shenzhen pool;  4 yuedong pool;
pool1 = dataset[dataset.pool_id==1]
pool2 = dataset[dataset.pool_id==2]
pool3 = dataset[dataset.pool_id==3]
pool4 = dataset[dataset.pool_id==4]


print (pool1)

X = pool1.iloc[:,1:2].values.astype(np.int64)/ 10 ** 9
#X = pd.Timestamp(X).values
y = pool1.iloc[:,2].values/1
print(X)
print(y)
#dataset = pd.read_csv('~/Desktop/Position_Salaries.csv')
# X = dataset.iloc[:, 1:2].values
# y = dataset.iloc[:, 2].values

# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
# lin_reg1 = LinearRegression()
# lin_reg1.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)


# Visualising the Linear Regression results
# plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg1.predict(X), color = 'blue')
# plt.title('Truth or Bluff (Linear Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

#print(X_poly)


# pridictTime = '2019-01-22 13:25:00'
# pridictTime = time.mktime(time.strptime(pridictTime,'%Y-%m-%d %H:%M:%S'))
# t = np.datetime64(pridictTime)

#use 3days timestamp to pridict usernumbers
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
yesterday_end_time = int(time.mktime(time.strptime(str(today), '%Y-%m-%d'))) - 1
today_start_time = yesterday_end_time + 1

print(today_start_time)
daysToCome = []
#863 = 3days * 60min / 5per
for i in range(0,864):
    #print (i)
    daysToCome.append(today_start_time+i*300)

#print(daysToCome)


df = pd.DataFrame({'future3ds':daysToCome})
print(df)
#print(type(df))

#change to matrix
t = df.values

#print(t)
pridictValues = lin_reg.predict(poly_reg.fit_transform(t))
print(pridictValues.tolist())

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.plot(t, lin_reg.predict(poly_reg.fit_transform(t)), color = 'red')
plt.title('sgw users shenzhen pool(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('sgwusers')

plt.show()




