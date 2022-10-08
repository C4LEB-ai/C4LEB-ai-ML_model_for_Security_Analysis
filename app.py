import re
import random
import uvicorn
import numpy as np
import pickle as pk
import pandas as pd
from typing import Union
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles



# Import Scikit-learn helper functions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Import Scikit-learn metric functions
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier


app = FastAPI()

#defining the tokenizer funtion.
def tokenizer(url):
  """Separates feature words from the raw data
  Keyword arguments:
    url ---- The full URL
    
  :Returns -- The tokenized words; returned as a list
  """
  
  # Split by slash (/) and dash (-)
  tokens = re.split('[/-]', url)
  

  for i in tokens:
    # Include the splits extensions and subdomains
    if i.find(".") >= 0:
      dot_split = i.split('.')
      
      # Remove .com and www. since they're too common
      if "com" in dot_split:
        dot_split.remove("com")
      if "www" in dot_split:
        dot_split.remove("www")
      
      tokens += dot_split
  return tokens
    

#Loading the models
model = pk.load(open("model.sav", 'rb'))
vectorizer = pk.load(open("vectorizer.sav", 'rb'))
#loading the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

#rendering the html templates
templates = Jinja2Templates(directory="templates")
@app.get("/page", response_class=HTMLResponse)
async def home(request: Request):
    page = templates.TemplateResponse('index.html', context ={'request': request})
    return page
  
#making predictions with the model
@app.get("/page", response_class=HTMLResponse)
async def homePost(url: str = Form()):
    list_1 = []
    url = str(url)

    list_1.append(url)
    testing_ = pd.DataFrame(data = list_1, columns = ['url'])
    test_count_d = vectorizer.transform(testing_['url'])
    result = model.predict(test_count_d)
    return f"\nThis url is: {result[-1]}"

#Creating the API
# @app.post("/page/api", response_class=HTMLResponse)
# async def homePost(url: str = Form()):
#     list_1 = []
#     url = str(url)

#     list_1.append(url)
#     testing_ = pd.DataFrame(data = list_1, columns = ['url'])
#     test_count_d = vectorizer.transform(testing_['url'])
#     result = model.predict(test_count_d)
#     return f"\nThis url is: {result[-1]}"

@app.post("/{param:path}", name="path-convertor")
async def testing(param):
    list_1 = []
    url = str(param)

    list_1.append(url)
    testing_ = pd.DataFrame(data = list_1, columns = ['url'])
    test_count_d = vectorizer.transform(testing_['url'])
    result = model.predict(test_count_d)
    return f"This url: {result[-1]}"


#the driver code
if __name__ == '__main__':
    uvicorn.run(app, host = "127.0.0.1", port = 8000)