import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df =pd.read_csv("/home/pi/python3/test/rForest/test_data.csv")
df.head()
#https://github.com/jadeyee/r2d3-part-1-data/blob/master/part_1_data.csv
#/home/pi/python3/test/rForest/test_data.csv


# separate final test data to check predictions

df_test = df.iloc[219:229]
# removing final test data from model dataframe
df = df.drop([219,220,221,222,223,224,225,226,227,228,229])

#Separating the variables
y = df["in_sf"]
x = df.drop("in_sf",axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, random_state=42)
x_train.shape, x_test.shape

rfc = RandomForestClassifier(random_state=42, max_depth=8,n_estimators=100, min_samples_leaf= 5)
rfc.fit(x_train,y_train)

#Finding the best paramters for our model with the help of grid search (GridSearchCV)

gs_rfc = RandomForestClassifier(random_state=40)

params = {
    'max_depth': [2,5,8,10,20],
    'min_samples_leaf': [2,5,10,15,25,50],
    'n_estimators': [5,15,30,60,80,100]
}

gs = GridSearchCV(estimator=gs_rfc,param_grid=params,cv = 4,verbose=1, scoring="accuracy")

gs.fit(x_train, y_train)

gs.best_score_
rfc_optimal = gs.best_estimator_
rfc_optimal

#the optimal parameters from the ones selected above were 
#RandomForestClassifier(max_depth=10, min_samples_leaf=2, n_estimators=60,random_state=40)

rfc_optimal.feature_importances_
p_df = pd.DataFrame({ "Variable": x_train.columns,"Importance": rfc_optimal.feature_importances_})
p_df.sort_values(by="Importance", ascending=True)

rfc_final = RandomForestClassifier(random_state=42, max_depth=8,n_estimators=60, min_samples_leaf= 2)
rfc_final.fit(x,y)


df_test2 = df_test.drop("in_sf",axis=1)
rfc_final.predict(x_test)

comparison = []
for i in rfc_final.predict(x_test):
    y_test.iloc[i] == rfc_final.predict(x_test)[i]
    comparison.append(y_test.iloc[i] == rfc_final.predict(x_test)[i])
print( comparison )

rfc_final.predict(df_test2)

# The values for the test array that go through coincides with the predictions for that group as seen in the above test.
# After running the final test with data that the model has not seen the predictions are wrong, it fits to the second category, apartemnts from San Francisco and it shold be half for each category
#array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=int64)
#futher parameter calibration is needed 
#We can discern that elevation, price_per_sqft and year_built are the predominant variables in determining the classification
