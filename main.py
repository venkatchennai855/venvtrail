from fastapi import FastAPI
from embedding import *
    
app = FastAPI()

@app.get("/")
def read_root():
    query = "How can I find top-rated mutual funds on the platform?"
    result = final(query)
    return {"answer": result[0],"Question_tokens":result[1],"answer_tokens":result[2]}
