import json
import time

import numpy as np
import pandas as pd
import redis
import requests
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

#Connecting to the Vector Database

try:

    
    client = redis.Redis(host='4.175.77.222', port=6379, decode_responses=True)
    client.ping()

    print("connceted successfuly bozo!")

except redis.ConnectionError as e:
    print(f"failed to conncect to The Database: {e}")


data = r'C:\Users\Briankechy\scripts\summarized_conversations.json'
try:
    with open(data, 'r') as file:
        loaded_data = json.load(file)
    print("Data Loaded Successfully")
    print(loaded_data)
except FileNotFoundError:
    print("Check your Filepath or Data")
except json.JSONDecodeError:
    print("error decoding JSON")

#defining a pipeline to upload Data
pipeline = client.pipeline()
for i, conversation in enumerate(data, start=1):
    redis_key = f"conversation:{i:o3}"
    pipeline.json().set(redis_key, "$", conversation)

res = pipeline.execute()
