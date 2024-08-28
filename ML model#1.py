#My First ML model
#data load:
import pandas as pd  
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")



#data seperation as X and Y:
y = df['logS']
x = df.drop('logS', axis=1)

 
 # data splitting:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=100)


# MODEL BUILDING:


     # lINEAR REGRESSION ALGO:
     
     
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) # training regresion model on data set

# applying the model to make a prediction:
y_lr_train_pred = lr.predict(x_train)
    #print(y_lr_train_pred)
y_lr_test_pred = lr.predict(x_test)
      #print(y_lr_test_pred)



#Evaluate Model Performance:
from sklearn.metrics import mean_squared_error, r2_score
lr_train_mee = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mee  = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
    #complicated way of current data evaluation:
    #print(lr_train_mee, lr_train_r2, lr_test_mee, lr_test_r2 )
    #better way is:
lr_results = pd.DataFrame(['Linear regression', lr_train_mee, lr_train_r2, lr_test_mee, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
#print(lr_results)
 
 
    # RANDOM FOREST ALGO:
    
# training the model:
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth = 2, random_state=100)
rf.fit(x_train, y_train)

#applying the model to make a prediction:
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# evaluate model performance:
from sklearn.metrics import mean_squared_error, r2_score
rf_train_mee = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mee  = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)
rf_results = pd.DataFrame(['Random forest', rf_train_mee, rf_train_r2, rf_test_mee, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
#print(rf_results)


       # model comparison/ algorithm comparison
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop = True)
print(df_models)


#DATA VISUALIZATION:

import matplotlib.pyplot as plt
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)

print(plt.plot())
plt.xlabel('Predict logS')
plt.ylabel('Experimental logS')
plt.show()
