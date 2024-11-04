# RAG-Powered Finance Guru: Your Smart AI Assistant

## Overview

This project implements a conversational AI assistant that utilizes Retrieval-Augmented Generation (RAG) techniques to provide finance-related answers based on user queries. The system combines various models, including GPT-2 and LLaMA, to generate responses, and it can retrieve information from local documents and Wikipedia.

## Features

- **Multi-Model Support**: Users can choose between LLaMA with RAG, GPT-2 with RAG, LLaMA, or GPT-2 for generating responses.
- **Real-Time Interaction**: Responses are generated in real-time, enhancing user engagement.
- **Dynamic Context Retrieval**: The system retrieves relevant information from both local knowledge bases and Wikipedia based on the current query.
- **Response Summarization**: Text summarization capabilities to handle large inputs and condense information for clearer responses.
- **User Feedback Mechanism**: Users can rate the quality of responses, which helps improve future interactions.
- **Conversation History**: The application maintains a history of conversations, allowing users to revisit previous queries and responses.
- **Automatic Knowledge Base Management**: The system initializes, saves, and loads a knowledge base from local documents for efficient querying.

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
   
2. **Install the required packages**:
   Make sure you have Python 3.8+ installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3. **Install Ollama**: Follow the installation instructions for [Ollama](https://ollama.com/docs/installation) to set up the local LLaMA model.

## Usage

1. **Run the Application**:
    To start the application, run:
    ```bash
    streamlit run app.py
    ```
   
2. **Interact with the Assistant**:
   - Enter your query in the input box.
   - Choose the model you want to use.
   - Submit your query to receive a response from the assistant.

3. **Provide Feedback**: After receiving a response, you can rate its quality, which helps improve the model's future performance.

## Code Structure

- **Model Initialization**: Loads necessary models and initializes the knowledge base.
- **Response Generation**: Implements functions for generating responses using either LLaMA or GPT-2, including handling context retrieval and similarity computations.
- **User Interaction**: The application utilizes Streamlit for the user interface, managing queries, responses, and conversation history.



