#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset
df=pd.read_csv('Student_Social_Media_And_Mental_Health_Impact.csv')

#Look towards the data
print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.info())
print(df.describe())


#EDA
#1.Distribution of the target
sns.histplot(df['Mental_Health_Score'],kde=True)
plt.show()

#2.Correlation heatmap
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

#3.Sress level vs mental health score
print(df["Stress_Level"].unique())
order=["Low","Medium","High","Very High"]
sns.boxplot(x="Stress_Level",y="Mental_Health_Score",data=df,order=order)
plt.show()

#4.Daily usage hours vs mental health score
sns.scatterplot(x='Avg_Daily_Usage_Hours',y="Mental_Health_Score",data=df,color="Red")
plt.show()

#5.Sleep hours vs mental health score
sns.scatterplot(x='Sleep_Hours_Per_Night',y="Mental_Health_Score",data=df)
plt.show()

#6.Most used platform(count)
platform_count=df["Most_Used_Platform"].value_counts()
plt.figure(figsize=(10,5))
sns.countplot(x="Most_Used_Platform",data=df,order=platform_count.index)
plt.show()

#Checking Outliers
num_feature=df.select_dtypes(include='number')
Q1=num_feature.quantile(0.25)
Q3=num_feature.quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR

outliers=(num_feature<lower_bound) | (num_feature>upper_bound)
print(outliers.sum())

#Data cleaning
#1.To drop duplicates
df=df.drop_duplicates()

#2.Converting negative value into releastic value
df["Physical_Activity_Hours"]=df["Physical_Activity_Hours"].clip(lower=0)
print(df.describe())

#Skewness
num_cols=df.select_dtypes(include="number")
print(num_cols.skew())

#Feature engineering
top_countries=df["Country"].value_counts().index[:10].tolist()
def group_countries(country):
    if country in top_countries:
        return country
    else:
        return 'Other'
df['Grouped_country']=df['Country'].apply(group_countries)

#Train-Test Split
from sklearn.model_selection import train_test_split
skews_cols=['Study_Hours']
other_num_cols=['Age','Avg_Daily_Usage_Hours','Daily_Unlocks','Physical_Activity_Hours','Sleep_Hours_Per_Night']
ordinal_col=['Stress_Level']
normal=['Gender','Academic_Level','Most_Used_Platform','Purpose_Of_Use','Grouped_country']
feature_col=skews_cols+other_num_cols+ordinal_col+normal
x=df[feature_col]
y=df['Mental_Health_Score']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Build a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

#1.Skewed features 
skew_pipeline=Pipeline(steps=[('log_transform',FunctionTransformer(np.log1p)),('scale',StandardScaler())])

#2.Numeric feature
plain_numeric_pipeline=Pipeline(steps=[('scale',StandardScaler())])

#3.Ordinal 
ordinal_pipeline=Pipeline(steps=[('encode',OrdinalEncoder(categories=[['Low','Medium','High','Very High']]))])

#4.Nominal features
nominal_pipeline=Pipeline(steps=[('encode',OneHotEncoder(handle_unknown='ignore'))])

preprocessor=ColumnTransformer(transformers=[('Skewed_pipeline',skew_pipeline,skews_cols),('Plain_numeric_pipeline',plain_numeric_pipeline,other_num_cols),('Ordinal_pipeline',ordinal_pipeline,ordinal_col),('Nominal_pipeline',nominal_pipeline,normal)])

#Model building
#1.Baseline:LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
lr_pipeline=Pipeline(steps=[('Preprocessor',preprocessor),('regression',LinearRegression())])

lr_pipeline.fit(x_train,y_train)
lr_preds=lr_pipeline.predict(x_test)
lr_preds_train=lr_pipeline.predict(x_train)
lr_r2_testing=r2_score(y_test,lr_preds)
print('Accuracy of testing',lr_r2_testing)
lr_r2_train=r2_score(y_train,lr_preds_train)
print('Accuracy of testing',lr_r2_train)
print('Mean Absolute Error:', mean_absolute_error(y_test, lr_preds))
print('Mean Squared Error:', mean_squared_error(y_test, lr_preds))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, lr_preds)))

#2.Random forest
from sklearn.ensemble import RandomForestRegressor
rf_pipeline=Pipeline(steps=[('preprocessor',preprocessor),('random forest',RandomForestRegressor(random_state=42))])
rf_pipeline.fit(x_train,y_train)
rf_preds=rf_pipeline.predict(x_test)
rf_preds_train=rf_pipeline.predict(x_train)
rf_r2_testing=r2_score(y_test,rf_preds)
print('Accuracy of testing',rf_r2_testing)
rf_r2_train=r2_score(y_train,rf_preds_train)
print('Accuracy of training',rf_r2_train)
MAE=mean_absolute_error(y_test,rf_preds)
print('MAE:',MAE)

#Hyperparameter tunning :the random forest
from sklearn.model_selection import RandomizedSearchCV
param_grid={'random forest__n_estimators':[100,200,300],'random forest__max_depth':[5,10,15],'random forest__min_samples_split':[2,5,10],'random forest__min_samples_leaf':[1,2,4]}
random_search=RandomizedSearchCV(estimator=rf_pipeline,param_distributions=param_grid,n_iter=15,cv=5,scoring='r2',random_state=42,n_jobs=-1)
random_search.fit(x_train,y_train)
random_search.best_params_
rf_best_pipeline=random_search.best_estimator_
rf_best_pred=rf_best_pipeline.predict(x_test)
print("R2 score:",r2_score(y_test,rf_best_pred))
print("MAE:",mean_absolute_error(y_test,rf_best_pred))

# Model Evaluation

models = {
    'Linear Regression': lr_pipeline,
    'Random Forest': rf_pipeline,
    'Tuned Random Forest': random_search.best_estimator_
}

results = []

for name, model in models.items():
    pred = model.predict(x_test)

    results.append({
        'Model': name,
        'R2 Score': r2_score(y_test, pred),
        'MAE': mean_absolute_error(y_test, pred),
        'MSE': mean_squared_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred))
    })

results_df = pd.DataFrame(results)
print(results_df.round(4))

# Best Model
print("\nBest Model:", results_df.loc[
    results_df['R2 Score'].idxmax(), 'Model'
])

# Visualization
sns.barplot(data=results_df, x='Model', y='R2 Score')
plt.xticks(rotation=15)
plt.title('Model R2 Score Comparison')
plt.tight_layout()
plt.show()

#Save the model
import joblib
joblib.dump(rf_pipeline,'Mental_Health_Model.pkl')