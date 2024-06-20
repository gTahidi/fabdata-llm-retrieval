# %%
#! %matplotlib qt
! pip install redis 
! pip install pandas 
! pip install sentence-transformers 
! pip install tabulate
! pip install sentence-transformers


#%%

#checking connection

import redis
import json

# Connect to Redis
client = redis.Redis(host="", port=6379, decode_responses=True)



# %%
# Upserting data
data_path = r''

# %%

# Load JSON data
from sentence_transformers import SentenceTransformer

try:
    with open(data_path, 'r') as file:
        loaded_data = json.load(file)
    print("Data Loaded Successfully")
except FileNotFoundError:
    print("Check your Filepath or Data")
    loaded_data = None
except json.JSONDecodeError:
    print("Error decoding JSON")
    loaded_data = None

if loaded_data:
    # Initialize the sentence transformer model
    model = SentenceTransformer('msmarco-distilbert-base-v4')

    # Define a pipeline to upload data
    pipeline = client.pipeline()

    # Iterate over the data and set Redis keys
    for key, conversations in loaded_data.items():
        for index, conversation in enumerate(conversations):
            # Construct a unique key for each conversation
            unique_key = f"{key}_{index}"
            # Generate embeddings for new data
            if 'summary_embedding' not in conversation:
                conversation['summary_embedding'] = model.encode(conversation['summary']).tolist()
            if 'intent_embedding' not in conversation:
                conversation['intent_embedding'] = model.encode(conversation['intent']).tolist()
            # Set the key in Redis
            pipeline.json().set(unique_key, "$", conversation)

    # Execute the pipeline
    try:
        res = pipeline.execute()
        print("Data Uploaded Successfully")
    except Exception as e:
        print(f"Error uploading data to Redis: {e}")
# %%
# Fetch an example entry
data = client.json().get('254757363543_4')
print(json.dumps(data, indent=2))

# %%
# creating schema
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


# Define the schema for the Redis search index
schema = (
    TextField("$.summary", no_stem=True, as_name="summary"),
    TextField("$.intent", no_stem=True, as_name="intent"),
    VectorField(
        "$.summary_embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": 768,  # Update the dimension based on actual data
            "DISTANCE_METRIC": "COSINE"
        },
        as_name="summary_vector"
    ),
    VectorField(
        "$.intent_embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": 768,  # Update the dimension based on actual data
            "DISTANCE_METRIC": "COSINE"
        },
        as_name="intent_vector"
    )
)

# Define the index definition for JSON documents
definition = IndexDefinition(prefix=[], index_type=IndexType.JSON)


# Drop the existing index if it exists
try:
    client.ft("idx_summa").dropindex(delete_documents=False)
    print("Index deleted successfully.")
except Exception as e:
    print(f"Failed to delete index: {str(e)}")

# Create the index
index_name = "idx_summa"
try:
    client.ft(index_name).create_index(schema, definition=definition)
    print("Index created successfully!")
except Exception as e:
    print(f"Error creating index: {str(e)}")

# Fetch and print index information
index_info = client.ft(index_name).info()
print(index_info)


# %%
# simple search query
# Simple search query
def basic_text_search(index_name, query):
    try:
        results = client.execute_command(
            'FT.SEARCH', index_name, query
        )
        return results
    except Exception as e:
        print(f"Error performing text search: {str(e)}")
        return []

# Example query to search for documents with "lesson plans" in the summary
query = 'programming'
search_results = basic_text_search(index_name, f'@summary:{query}')

# Process and print results
print("Text Search Results:")
for result in search_results:
    print(result)


# %%
# List all keys in the Redis database to see what's available
all_keys = client.keys("*")
print(all_keys)

# %%
# vector search function
import struct
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('msmarco-distilbert-base-v4')

def convert_vector_to_binary(vector):
    # Convert a list of floats into a binary string
    return struct.pack(f'{len(vector)}f', *vector)

def knn_search(client, index_name, vector_field, query_vector, num_neighbours):
    # Convert the query vector to binary format
    query_vector_binary = convert_vector_to_binary(query_vector)
    
    # Prepare the KNN search query
    query = f"(*)=>[KNN {num_neighbours} @{vector_field} $vector]"
    params = ["PARAMS", "2", "vector", query_vector_binary]

    # Execute the search query
    results = client.execute_command(
        'FT.SEARCH', index_name, query, *params, "RETURN", "3", vector_field, "summary", "intent", "DIALECT", "2"
    )
    return results
# %%

# Test the search for summary embeddings
summary_vector = model.encode("lesson planning.").tolist()
summary_results = knn_search(client, 'idx_summa', 'summary_vector', summary_vector, 3)
print("Summary Search Results:", summary_results)

# Test the search for intent embeddings
intent_vector = model.encode("exploring new teaching methods.").tolist()
intent_results = knn_search(client, 'idx_summa', 'intent_vector', intent_vector, 3)
print("Intent Search Results:", intent_results)

# %%
# testing relevance of the result
def test_query_relevance(client, index_name, query_vector, expected_docs):
    results = knn_search(client, index_name, 'intent_vector', query_vector, 3)
    print("Raw Results:", results)  # Debug print

    # Check if results list is non-empty and has an even number of elements following the count
    if isinstance(results, list) and len(results) > 1 and (len(results) - 1) % 2 == 0:
        retrieved_docs = []
        for i in range(1, len(results), 2):  # Start from 1 and increment by 2 to skip over IDs
            fields = results[i+1]  # fields is the list following the ID
            doc_info = {fields[j]: fields[j + 1] for j in range(0, len(fields), 2)}  # Create a dictionary from the list
            retrieved_docs.append(doc_info.get('summary', 'No summary'))  # Get 'summary' field, default to 'No summary'
    else:
        retrieved_docs = []

    print("Retrieved Docs:", retrieved_docs)
    print("Expected Docs:", expected_docs)
    # Calculate relevance
    relevance_score = len(set(retrieved_docs) & set(expected_docs)) / len(set(expected_docs)) if expected_docs else 0
    print("Relevance Score:", relevance_score)

# Example usage
example_vector = model.encode("importance of lesson planning in education").tolist()
expected_results = ["Lesson planning guide", "Educational strategies", "Teaching methods"]
test_query_relevance(client, 'idx_summa', example_vector, expected_results)


# %%
# parsing KNN results
import ast

def parse_knn_results(results):
    documents = []
    # Skip the first item (number of results)
    for i in range(1, len(results), 2):
        doc_id = results[i]
        data = results[i + 1]
        doc_data = {
            'id': doc_id,
            'summary': data[data.index('summary') + 1],
            'vector': ast.literal_eval(data[data.index('summary_vector') + 1] if 'summary_vector' in data else data[data.index('intent_vector') + 1])
        }
        documents.append(doc_data)
    return documents



# %%
# MMR
import numpy as np



def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def mmr_score(selected_docs, candidate_doc, query_vec, lambda_val=0.5):
    """Calculate MMR score for a single candidate document."""
    doc_vec = np.array(candidate_doc['vector'])
    relevance = cosine_similarity(doc_vec, np.array(query_vec))
    max_similarity = max([cosine_similarity(doc_vec, np.array(selected_doc['vector'])) for selected_doc in selected_docs] or [0])
    return lambda_val * relevance - (1 - lambda_val) * max_similarity

def select_documents_with_mmr(docs, query_vec, lambda_val=0.7, top_k=5):
    """Select top K documents using MMR."""
    selected_docs = []
    remaining_docs = list(docs)  # Create a mutable copy of docs

    while remaining_docs and len(selected_docs) < top_k:
        mmr_scores = {doc['id']: mmr_score(selected_docs, doc, query_vec, lambda_val) for doc in remaining_docs}
        selected_doc_id = max(mmr_scores, key=mmr_scores.get)
        selected_doc = next(doc for doc in remaining_docs if doc['id'] == selected_doc_id)
        selected_docs.append(selected_doc)
        remaining_docs.remove(selected_doc)
    
    return selected_docs

# testing 
def test_mmr_integration(query):
    # Encode query to vector
    query_vector = model.encode(query).tolist()

    # Perform KNN search
    knn_results = knn_search(client, 'idx_summa', 'summary_vector', query_vector, 10)  # Adjust based on field
    
    # Parse results
    parsed_results = parse_knn_results(knn_results)

    # Apply MMR
    selected_docs = select_documents_with_mmr(parsed_results, query_vector, lambda_val=0.75, top_k=3)

    # Print selected documents with MMR
    print("Selected Documents with MMR:")
    for doc in selected_docs:
        print(f"ID: {doc['id']}, Summary: {doc['summary']}")

# Example usage
test_mmr_integration("importance of lesson planning in education")
