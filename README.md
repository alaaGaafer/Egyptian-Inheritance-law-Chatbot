# Inheritance Law Question Answering System

## Overview

The **Inheritance Law Question Answering System** is a web-based application designed to provide answers to questions related to inheritance laws based on a given legal document. The system utilizes Retrieval-Augmented Generation (RAG) techniques to retrieve relevant information from a PDF document and generate precise answers. It leverages advanced language models for natural language understanding and question answering.

## Project Description

This project focuses on automating the process of answering questions related to inheritance laws. The main features include:

- **PDF Document Processing**: The system downloads and processes PDF documents containing legal information about inheritance laws.
- **Text Splitting**: The document is split into manageable chunks of text for efficient processing.
- **Embedding and Retrieval**: Text chunks are embedded using OpenAI's language model embeddings and stored in a Chroma database for efficient retrieval.
- **Question Answering**: Users can ask questions about inheritance, and the system retrieves relevant context from the document and generates concise answers using language models.
- **Interactive Chatbot**: An interactive chatbot interface allows users to ask questions and receive answers in a conversational format.

## Dependencies

The project relies on several key dependencies, which are detailed below:

- **LangChain**: A framework for managing language model interactions and building complex chains.
  - `langchain_openai`
  - `langchain_chroma`
  - `langchain_core`

- **OpenAI**: For utilizing OpenAI's language models and embeddings.
  - `openai`

- **PyPDF2**: For reading and extracting text from PDF documents.
  - `PyPDF2`

- **Requests**: For handling HTTP requests to download PDF documents.
  - `requests`

- **NLTK**: For natural language processing, specifically sentence tokenization.
  - `nltk`

- **Streamlit**: For creating a web-based user interface .
  - `streamlit`

- **Dotenv**: For loading environment variables from a `.env` file.
  - `python-dotenv`



