import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_insights(results_df):
    prompt = f''' You are a data scientist and you are given a dataframe of model results:
    {results_df.to_dict(orient='records')}
    1. Identify the best model.
    2. Explain why is it the best model based on appropriate metric.
    3. Summarise the performance of that model.
    '''
    response = model.generate_content(prompt)
    return response.text

def suggest_improvements(results_df):
    prompt = f''' You are a data scientist and you are given a dataframe of model results:
    {results_df.to_dict(orient='records')}
    Suggest:
    1. Ways to improve model performance.
    2. Better algorithms to use for the given dataset and problem type.
    3. Better preprocessing techniques to use for the given dataset and problem type.
    '''
    response = model.generate_content(prompt)
    return response.text
