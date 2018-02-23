import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

price = []
date = []

def get_data_from(filename):
    file = filename +'.csv'
    stock = pd.read_csv(file)
    iters = len(stock)
    for i in range(0,iters):
        tdate = int(stock['Date'][i].split('-')[0])
        date.append(tdate)
        tprice = float(stock['Open'][i])
        price.append(tprice)
    price.reverse()
    date.reverse()
    return price,date

'''
def get_new_price(price, data, x):
    datex = []
    datex = np.reshape(date,(len(date),1))

    svr_lin = SVR(kernel ='linear', C =1e3)
    svr_pol = SVR(kernel ='poly',   C =1e3, degree =2)
    svr_rbf = SVR(kernel ='rbf',    C=1e3,  gamma = 0.1)

    svr_lin.fit(datex,price)
    svr_pol.fit(datex,price)
    svr_rbf.fit(datex,price)
    
    svr_rbf_pred = svr_rbf.predict(datex)

    dataset = pd.read_csv('googl.csv')
    nofstocks = len(dataset)
    stockdates = []
    for i in range(0,nofstocks):
        sdate = int(dataset['Date'][i].split('-')[0])
        stockdates.append(sdate)
    X = stockdates
    y = svr_rbf_pred
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
   
    y_pred = classifier.predict(X_test)
    
    # Visualising the Test set results
    plt.scatter(datex,price, color='black', label='Data')
    #plt.plot(datex, svr_lin.predict(datex), color = 'orange', label ='Linear SVR')
    #plt.plot(datex, svr_pol.predict(datex), color = 'blue',   label = 'Polynomial SVR')
    plt.plot(X, y_pred, color = 'green',  label = 'Radial Basis Func SVR')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
'''

def get_price(price,date,x):
    totaldate = []
#    for i in range(1, x+1):
#        totaldates[i] = i
    totaldate = range(1,x+1)
    totaldates = np.reshape(totaldate,(len(totaldate),1))
    datex = []
    datex = np.reshape(date,(len(date),1))

    svr_lin = SVR(kernel ='linear', C =1e3)
    svr_pol = SVR(kernel ='poly',   C =1e3, degree =2)
    svr_rbf = SVR(kernel ='rbf',    C=1e3,  gamma = 0.1)

    svr_lin.fit(datex,price)
    svr_pol.fit(datex,price)
    svr_rbf.fit(datex,price)

    svr_rbf_pred = svr_rbf.predict(datex)
        
#    svr_rbf.predict(40)
    new_pred = svr_rbf.predict(totaldates)
    
    plt.scatter(datex,price, color='black', label='Data')
    plt.plot(datex, svr_lin.predict(datex), color = 'orange', label ='Linear SVR')
    plt.plot(datex, svr_pol.predict(datex), color = 'blue',   label = 'Polynomial SVR')
#    plt.plot(datex, svr_rbf_pred, color = 'green',  label = 'Radial Basis Func SVR')
    plt.plot(totaldates, new_pred, color = 'green',  label = 'Radial Basis Func SVR')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0], svr_pol.predict(x)[0], svr_rbf.predict(x)[0]

get_data_from('googl')
get_price(price, date, 40)
#get_new_price(price, date, 30)