## Introduction

FabData-LLM-Retrieval is an end-to-end solution for building a Retrieval-Augmented Generation (RAG) system from a document catalogue. It handles: 

- Content extraction and management
    - Text extraction
    - Content encoding
        - Chunking at multiple configurable sizes
        - Chunk embeddings
            - Currently only supports OpenAI ada-002 embeddings
    - Content interpretation
        - Automatic extraction of references
        - Automatic web-scraping of references
        - Visualistion of document semantic space
        - Clustering of documents
        - Automatic document tagging
    - Datastructures to help with content management and storage

- Vector database integration
    - Currently supports Redis with the Json and Search modules
    - Automated TLS-enabled database deployment and persistent storage on an Azure container instance, via our sister project: [Redis Stack Server ACI](https://github.com/AI-for-Education/redis-stack-server-ACI)
    - Data upload
    - Semantic search
    - Filter by tags, chunk size
    - Result cleaning (e.g. duplicate removal)
    - Easy instantiation of content datastuctures from existing database

- Full chatbot implementation
    - Integrated with our other sister project: [FabData-LLM](https://github.com/AI-for-Education/fabdata-llm)
    - RAG chatbot implemented via tool-use frameworks
    - Supports all tool-use enabled models from OpenAI and Anthropic
    - Tools take advantage of extra information, such as references, supporting documents, tags, chunk sizes, as well as the standard semantic search
    - Tools are chainable, which can lead to some intelligent behaviour from the system without any additional user prompting (for example searching for small chunks first and then expanding to larger chunks for more detailed results, automatically improving searches with document ID filters if the returned material is not relevant, choosing to search through all the references of a specfic document to improve the result)