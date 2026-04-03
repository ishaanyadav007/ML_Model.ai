import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

#model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor

#metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, r2_score,root_mean_squared_error

# To get ai insights from function
from analysis import generate_insights,suggest_improvements

st.set_page_config(page_title="AI ML Insights", page_icon=":robot:", layout="wide")

st.title("Auto ML + AI Insights ֎")
st.subheader(":green[To learnt the given data and to fit the ML models and to get AI Insights using Gemini.]")
file = st.file_uploader("Upload your csv file here: ", type=["csv", "xlsx", "txt"], help="Upload your dataset to be used for training the models.")

if file:
    df = pd.read_csv(file)
    st.write('### Data Preview')
    st.dataframe(df.head())
    st.write('### Data Information')
    st.write(df.info())
    st.write('### Data Description')
    st.write(df.describe())
    st.write('### Data Columns')
    st.write(df.columns)
    st.write('### Data Types')
    st.write(df.dtypes)
    
    target = st.selectbox("Select the target column: ", df.columns)
    if target:
        st.write(f"### Target Column: {target}")
        st.write(f"### Target Column data type: {df[target].dtype}")
        st.write(f"### Target Column Values: {df[target].unique()}")
        st.write(f"### Target Column Value Count: {df[target].value_counts()}")

        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        #preprocessing
        num = X.select_dtypes(include=['number']).columns.tolist()
        cat = X.select_dtypes(include=['object']).columns.tolist()
        
        
        X[num] = X[num].fillna(X[num].median())
        X[cat] = X[cat].fillna(X[cat].mode().iloc[0])
        
        
        # encoding
        X = pd.get_dummies(X, columns=cat, drop_first=True,dtype=int)

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # detect the problem type
        if y.dtype == 'object' or len(np.unique(y)) <= 15:
            problem_type = 'Classification'
        else:
            problem_type = 'Regression'
        
        st.write(f"### Problem Type: {problem_type} ⚠️")
        
        # split the data
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
         
        scaler = StandardScaler()
        xtrain[num] = scaler.fit_transform(xtrain[num])
        xtest[num] = scaler.transform(xtest[num])

        # MODELS
        results = []
        if problem_type == 'Classification':
            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
            }

            for name, model in models.items():
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)
                yprob = model.predict_proba(xtest)
                
                results.append({
                    'Model_name': name,
                    'Accuracy': accuracy_score(ytest, ypred),
                    'Precision': precision_score(ytest, ypred,average='weighted'),
                    'Recall': recall_score(ytest, ypred,average='weighted'),
                    'F1 Score': f1_score(ytest, ypred,average='weighted'),
                    'ROC AUC': roc_auc_score(ytest, yprob ,multi_class='ovr')
                })
        
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            } 
            
            for name, model in models.items():
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)
                
                results.append({
                    'Model_name': name,
                    'R2 Score':r2_score(ytest, ypred),
                    'RMSE Score': root_mean_squared_error(ytest, ypred)
                })

        results_df = pd.DataFrame(results)
        st.write("### :red[Model Results 📝]")
        st.dataframe(results_df)

        # best model
        if problem_type == 'Classification':
            best_model = results_df.loc[results_df['Accuracy'].idxmax()]
        else:
            best_model = results_df.loc[results_df['R2 Score'].idxmax()]
        st.write("### :green[Best Model 🎯]")
        st.write(best_model)

        # for visualisation purpose
        if problem_type == 'Regression':
            st.bar_chart(results_df.set_index('Model_name')['R2 Score']) 
# Streamlit interprets st.bar_chart(df)
# Index → X-axis
# Columns → Y values
            st.bar_chart(results_df.set_index('Model_name')['RMSE Score'])
        else:
            st.bar_chart(results_df.set_index('Model_name')['Accuracy'])
            st.bar_chart(results_df.set_index('Model_name')['F1 Score'])

        
        # AI INSIGHTS:

        if st.button(':blue[Generate_Summary]'):
            summary = generate_insights(results_df)
            st.write(summary)
        
        if st.button(':blue[Suggest_improvements]'):
            improvements = suggest_improvements(results_df)
            st.write(improvements)

        # DOWNLOAD results_df as csv
        results_csv  = results_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download csv here',results_csv,"model_results.csv")
        

                 
        
        
             
            












