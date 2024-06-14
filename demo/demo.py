# %%
#! %matplotlib qt
! pip install redis 
! pip install pandas 
! pip install sentence-transformers 
! pip install tabulate


#%%

import redis
import json

# Connect to Redis
client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Fetch an example entry
data = client.json().get('summarized_conversations')
print(json.dumps(data, indent=2))


# %%

from sentence_transformers import SentenceTransformer
# Initialize the sentence transformer model

data = client.json().get('summarized_conversations')

# to do update model to use text davinci
# Initialize the sentence transformer model
model = SentenceTransformer('msmarco-distilbert-base-v4')

# Iterate through each user ID in the data

for user_id, conversations in data.items():
    for conversation in conversations:
        # Generate embeddings for both summary and intent
        summary_embedding = model.encode(conversation['summary']).tolist()
        intent_embedding = model.encode(conversation['intent']).tolist()
        
        # Append embeddings to the conversation entry
        conversation['summary_embedding'] = summary_embedding
        conversation['intent_embedding'] = intent_embedding

client.json().set('summarized_conversations', '$', data)



# %%
# Set up the connection to Redis
client = redis.Redis(host="4.175.77.222", port=6379, decode_responses=True)

# Fetch the data from Redis
data = client.json().get('summarized_conversations')

# Print the data in a formatted way
print(json.dumps(data, indent=2))


# %%

# Attempt to delete the existing index if it exists
try:
    client.ft("idx_summarized_conversations").dropindex(delete_documents=False)
    print("Index deleted successfully.")
except Exception as e:
    print(f"Failed to delete index: {str(e)}")


# %%
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect to Redis
client = redis.Redis(host='4.175.77.222', port=6379, decode_responses=True)

# Define the schema for the Redis search index
schema = (
    TextField("$.summary", no_stem=True, as_name="summary"),
    TextField("$.intent", no_stem=True, as_name="intent"),
    VectorField(
        "$.summary_embedding",
        "FLAT",
        attributes={
            "TYPE": "FLOAT32",
            "DIM": 768,  # Update the dimension based on actual data
            "DISTANCE_METRIC": "COSINE"
        },
        as_name="summary_vector"
    ),
    VectorField(
        "$.intent_embedding",
        "FLAT",
        attributes={
            "TYPE": "FLOAT32",
            "DIM": 768,  # Update the dimension based on actual data
            "DISTANCE_METRIC": "COSINE"
        },
        as_name="intent_vector"
    )
)

# Define the index definition for JSON documents
definition = IndexDefinition(prefix=["summarized_conversations:"], index_type=IndexType.JSON)

# Drop the existing index if it exists
try:
    client.ft("idx_summarized_conversations").dropindex(delete_documents=False)
    print("Index deleted successfully.")
except Exception as e:
    print(f"Failed to delete index: {str(e)}")

# Create the index
index_name = "idx_summarized_conversations"
try:
    client.ft(index_name).create_index(schema, definition=definition)
    print("Index created successfully!")
except Exception as e:
    print(f"Error creating index: {str(e)}")

# Fetch and print index information
index_info = client.ft(index_name).info()
print(index_info)

# %%
# Data insertion example
data = {
    "254757363543": [
        {
            "summary": "The first user inquired about Kubernetes Secrets and ConfigMaps, seeking a simple explanation of their functionalities. The second user engaged in a counting activity with the bot, successfully counting to ten and attempting to continue the count. The bot offered assistance for further activities or discussions.",
            "intent": "User 1: Seeking information on Kubernetes Secrets and ConfigMaps. User 2: Engaging in a counting activity and exploring further interactions with the bot.",
            "summary_embedding": [
                -0.24152256548404696, 0.14457350969314575, 0.27285557985305786
            ],
            "intent_embedding": [
                -0.2608829736709595, 0.703132152557373
            ]
        }
    ]
}

client.json().set('summarized_conversations', '$', data)
print("Data inserted successfully.")

# %%

from redis.commands.search.query import Query

# Define a simple search query to test the index
query = Query("@summary:Kubernetes")

try:
    results = client.ft("idx_summarized_conversations").search(query)
    print(f"Found {results.total} results:")
    for doc in results.docs:
        print(f"Summary: {doc.summary}, Intent: {doc.intent}")
except Exception as e:
    print(f"Error executing query: {str(e)}")


# %%

# confirm proper insertions
import json

data = client.json().get('summarized_conversations')
print(json.dumps(data, indent=2))


# %%
# Fetch and print index information
index_info = client.ft("idx_summarized_conversations").info()
print(index_info)

# %%
# Example simple query to test if the index works
results = client.ft("idx_summarized_conversations").search("@summary:kubernetis")
print("Number of results:", results.total)
for doc in results.docs:
    print(doc.summary)


# %%
import redis 
from redis.commands.search.query import Query
# Define the query
query = Query("@summary:(information)").paging(0, 5)  # Searches for "information" in the summary field

# Execute the search
try:
    results = client.ft("idx_summarized_conversations").search(query)
    print(f"Found {results.total} results:")
    for doc in results.docs:
        print(f"Summary: {doc.summary}, Intent: {doc.intent}")
except Exception as e:
    print(f"Error executing query: {str(e)}")


# %%
from sentence_transformers import SentenceTransformer
import numpy as np
import redis
from redis.commands.search.query import Query

# Initialize the sentence transformer model
model = SentenceTransformer('msmarco-distilbert-base-v4')

# Example query
query = "Information on educational content in a class scenario"
query_embedding = model.encode(query)
len(query)

query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

# Define the KNN search query
search_query = Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]').sort_by('vector_score').return_fields('vector_score', 'summary', 'intent').dialect(2)

try:
    # Perform the search with proper PARAMS
    response = client.ft("idx_summarized_conversations").search(search_query, query_params={'vec': query_vector})
    print(f"Query executed. Number of results: {len(response.docs)}")
    for doc in response.docs:
        print(f"Summary: {doc.summary}, Intent: {doc.intent}, Score: {doc.vector_score}")
except Exception as e:
    print(f"Error executing query: {e}")
# %%
#verify data insertion
keys = client.keys("summarized_conversations:*")
print(keys)  # Check if this returns any keys.


# %%
# Example to fetch the structure of one entry
entry = client.json().get('summarized_conversations:1')  # Adjust the key based on actual keys in your Redis
print(json.dumps(entry, indent=2))


# %%
# List all keys in the Redis database to see what's available
all_keys = client.keys("*")
print(all_keys)

# %%
# Create a search query targeting the 'summary' field with wildcard to match any content
# Using a broad matching pattern to ensure some results are returned
search_query = Query("@summary:(*)")

# Execute the search
try:
    results = client.ft("idx_summarized_conversations").search(search_query)
    if results.total > 0:
        print(f"Found {results.total} results:")
        for doc in results.docs:
            print(f"Summary: {doc.summary}, Intent: {doc.intent}")
    else:
        print("No results found.")
except Exception as e:
    print(f"Error executing query: {e}")


# %%
# checking key prefixes

keys = client.keys("*")  # Lists all keys in the database to understand the actual stored pattern.
print(keys)

# %%
# trying a different set of data (gtahidi_data)
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect to Redis
client = redis.Redis(host='4.175.77.222', port=6379, decode_responses=True)

# Define the schema for the Redis search index
schema = (
    TextField("$.gTahidi_Data[*].messages[*].content", no_stem=True, as_name="content"),
    TextField("$.gTahidi_Data[*].messages[*].role", no_stem=True, as_name="role")
)

# Define the index definition for JSON documents
definition = IndexDefinition(prefix=["gTahidi_Data"], index_type=IndexType.JSON)

# Create the index
index_name = "idx_gTahidi_Data"
try:
    client.ft(index_name).create_index(schema, definition=definition)
    print("Index created successfully!")
except Exception as e:
    print(f"Error creating index: {str(e)}")

# %%
# Perform a simple search to find messages with the word "website"
import redis
from redis.commands.search.query import Query

# Assuming you've already set up the Redis client and connected as shown previously
try:
    # Perform the search query
    query_result = client.ft("idx_gTahidi_Data").search("@content:example")
    # If the query executes successfully, proceed to check results
    if query_result.total == 0:
        print("No results found.")
        index_info = client.ft("idx_gTahidi_Data").info()
        print("Index Info:", index_info)
    else:
        # Print the contents if results exist
        for doc in query_result.docs:
            print(doc.content, doc.role)
except Exception as e:
    # If an error occurs during the search or fetching index info
    print(f"Error during search or fetching index info: {str(e)}")
    if 'query_result' in locals():
        # Check if 'query_result' is defined to avoid NameError
        if query_result.total == 0:
            index_info = client.ft("idx_gTahidi_Data").info()
            print("Index Info:", index_info)


# %%
# Fetch and print the entire JSON data for 'gTahidi_Data'
data = client.json().get('gTahidi_Data')
print(json.dumps(data, indent=4))

# %%
