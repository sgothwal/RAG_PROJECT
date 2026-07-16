from ragas import evaluate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable
from google import genai
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import pandas as pd
import time
import random
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.pipeline import retrieval,rerank,genereate
load_dotenv()

def load_eval_dataset()->list:
    try:
        base_path = Path(__file__).parent
        with open(base_path.parent / 'json_files' /'eval_dataset.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: The file final_data.json does not exist.")  

    return data   

def send_eval_chunk(data)->list:
    eval_dataset = []
    for item,i  in enumerate(data):
        print(f"Running: {item['question'][:60]}")
        result = eval_chunk(item)
        eval_dataset.append(result)
        time.sleep(4)
   
    return eval_dataset    

@traceable(name="RAG Evaluation Pipeline")
def eval_chunk(item)->dict:
   
    chunks = retrieval(item["question"])
    reranked = rerank(chunks, item["question"])
    answer = genereate(reranked, item["question"])
    
    return {
        "question": item["question"],
        "answer": answer,
        "contexts": [c[0].page_content for c in reranked],
        "ground_truth": item["ground_truth"],
        "source":item['source'],
        "page_number":item['page_number']
    }

def run_eval(eval_dataset)->list:
    test_cases = [
    LLMTestCase(
        input=item["question"],
        actual_output=item["answer"],
        retrieval_context=item["contexts"],
        expected_output=item["ground_truth"]
    )
    for item in eval_dataset
]
    metrics = [
    FaithfulnessMetric(threshold=0.7, model='gpt-4.1-mini'),
    AnswerRelevancyMetric(threshold=0.7, model='gpt-4.1-mini'),
    ContextualPrecisionMetric(threshold=0.7, model='gpt-4.1-mini'),
    ContextualRecallMetric(threshold=0.7, model='gpt-4.1-mini')
]
    result=evaluate(test_cases=test_cases,metrics=metrics)
    return result

def run():
    data=load_eval_dataset()
    eval_dataset=send_eval_chunk(data)
    result=run_eval(eval_dataset)
if __name__== "__main__":
    run()    

    



