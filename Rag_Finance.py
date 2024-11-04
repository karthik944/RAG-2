import os
import json
import streamlit as st
import subprocess
import re
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from wikipediaapi import Wikipedia
from collections import deque
import PyPDF2

# Setup the Sentence Embeddings Model
model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

# Setup Wikipedia API
wiki = Wikipedia('RAGBot/Rag-2', 'en')

# Load GPT-2 Model and Tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
g2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Setup the Summarization Pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text
def summarize_text(text, max_length=200):
    if len(text) > 1000:
        text = text[:1000]
    summaries = summarizer(text, max_length=max_length, truncation=True)
    return summaries[0]['summary_text']

# Function to preprocess the prompt
def preprocess_prompt(prompt: str) -> str:
    prompt = prompt.lower().strip()
    prompt = re.sub(r'[^\w\s]', '', prompt)
    return prompt

# Function to post-process the response
def postprocess_response(response: str) -> str:
    response = re.sub(r'<\|endoftext\|>', '', response)
    return response.strip()

# Function to generate text using local LLaMA model via Ollama
def generate_llama_response(prompt):
    process = subprocess.run(
        ['ollama', 'run', 'llama3.2:latest'],
        input=prompt,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if process.returncode != 0:
        error_message = process.stderr.strip()
        return f"Error running Ollama: {error_message}"
    
    response = process.stdout.strip()
    return response

# Function to generate text using GPT-2
def generate_gpt2_response(prompt, max_new_tokens=50, temperature=0.7, top_k=50, top_p=0.95):
    try:
        inputs = g2_tokenizer.encode(prompt, return_tensors="pt")
        max_input_length = 900

        if inputs.size(1) > max_input_length:
            st.warning("Input prompt exceeded the maximum length and was truncated.")
            inputs = inputs[:, -max_input_length:]

        outputs = gpt2_model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            do_sample=True,
            early_stopping=True
        )
        
        response = g2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return f"Error during generation: {str(e)}"

# Function to read PDF files from a directory
def read_pdfs_from_directory(directory):
    content_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                content_list.append(text)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return content_list

# Optimized Wikipedia search function
def search_wikipedia(query):
    stop_words = {"what", "who", "how", "is", "was", "are", "which", "when", "where", "why", "did", "do", "does"}
    filtered_words = [word for word in query.split() if word.lower() not in stop_words]
    search_phrase = " ".join(filtered_words)

    doc = wiki.page(search_phrase).text
    if not doc:
        return None
    
    paragraphs = doc.split('\n\n')
    return paragraphs

# Function to save knowledge base to a file
def save_knowledge_base(knowledge_base, filename='knowledge_base.json'):
    serializable_kb = {k: v.tolist() for k, v in knowledge_base.items()}
    with open(filename, 'w') as f:
        json.dump(serializable_kb, f)

# Function to load knowledge base from a file
def load_knowledge_base(filename='knowledge_base.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            loaded_kb = json.load(f)
            return {k: np.array(v) for k, v in loaded_kb.items()}
    return {}

# Function to initialize the knowledge base
def initialize_knowledge_base():
    books_directory = "Books"
    papers_directory = "Papers"

    books_content = read_pdfs_from_directory(books_directory)
    papers_content = read_pdfs_from_directory(papers_directory)

    all_content = books_content + papers_content
    knowledge_base = {}
    
    for text in all_content:
        summarized_content = summarize_text(text)
        knowledge_base[summarized_content] = model.encode(summarized_content)

    save_knowledge_base(knowledge_base)

# Function to compute similarity between query and context memory
def compute_similarity(query_embedding, context_memory):
    context_embeddings = model.encode([memory['user'] for memory in context_memory], normalize_embeddings=True)
    similarities = np.dot(context_embeddings, query_embedding.T).flatten()
    return similarities

# Retrieve relevant content from Wikipedia and local sources
def retrieve_relevant_content(query):
    query_embedding = model.encode(query, normalize_embeddings=True)
    local_content = list(st.session_state.knowledge_base.keys())
    local_embeddings = list(st.session_state.knowledge_base.values())
    wiki_paragraphs = search_wikipedia(query)

    if wiki_paragraphs:
        wiki_embeddings = model.encode(wiki_paragraphs, normalize_embeddings=True)
        combined_paragraphs = local_content + wiki_paragraphs
        combined_embeddings = np.vstack([local_embeddings, wiki_embeddings])
    else:
        combined_paragraphs = local_content
        combined_embeddings = np.vstack(local_embeddings)

    similarities = np.dot(combined_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-3:][::-1]
    top_docs = [combined_paragraphs[i] for i in top_indices]
    
    return top_docs

# Retrieve relevant context from memory based on the current query
def retrieve_dynamic_context(query):
    query_embedding = model.encode(query, normalize_embeddings=True)
    context_memory = list(st.session_state.context_memory)
    
    if not context_memory:
        return []

    similarities = compute_similarity(query_embedding, context_memory)
    # Get indices of the top N most relevant interactions (e.g., top 2)
    top_indices = np.argsort(similarities)[-2:][::-1]
    
    return [context_memory[i] for i in top_indices]

# Combining retrieval and generation (RAG workflow)
def rag_response(query, model_choice='llama'):
    start_time = time.time()

    # Retrieve dynamic context from memory
    dynamic_context = retrieve_dynamic_context(query)
    context = retrieve_relevant_content(query)

    if not context and not dynamic_context:
        return "I couldn't find relevant information."

    # Combine both dynamic context and retrieved content
    context_text = "\n".join([memory['chatbot'] for memory in dynamic_context] + context)

    prompt = f"""
    You are a financial expert tasked with analyzing and interpreting data to answer a question. Below is the information available:

    {context_text}

    Given this information, please provide a clear, detailed answer to the following finance-related question:

    {query}

    Ensure that your answer includes relevant financial concepts, key metrics, and, if applicable, a brief explanation of the reasoning behind the answer. If the information provided is insufficient to answer accurately, respond with "The provided information is insufficient to answer this question comprehensively."
"""

    if model_choice == 'LLaMA with RAG':
        response = generate_llama_response(prompt)
    elif model_choice == 'GPT-2 with RAG':
        response = generate_gpt2_response(prompt)
    else:
        if model_choice == 'LLaMA':
            response = generate_llama_response(query)
        else:
            response = generate_gpt2_response(query)

    end_time = time.time()
    response_time = end_time - start_time

    return response, response_time

st.title("RAG-Powered Finance Guru: Your Smart AI Assistant")

# Initialize session state variables if they don't exist
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = load_knowledge_base()
    if not st.session_state.knowledge_base:
        initialize_knowledge_base()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'context_memory' not in st.session_state:
    st.session_state.context_memory = deque(maxlen=5)  # Store the last 5 interactions

tab1, tab2 = st.tabs(["Chat", "Conversation History"])

 
# Function to handle feedback submission
def handle_feedback(previous_chat):
    rating_options = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
    
    # Create a dropdown for rating
    selected_rating = st.selectbox("Rate this response:", rating_options, key=f"rating_{previous_chat['user']}")
    
    # Create a button for submitting feedback
    if st.button("Submit Rating", key=f"submit_rating_{previous_chat['user']}"):
        # Store the rating in chat history
        previous_chat['rating'] = selected_rating
        # Clear the response display
        st.session_state.current_response = None
        st.success("Thank you for your feedback!")

# UI Code
with tab1:
    query = st.text_input("Your Query:", key="query_input")

    # Select Model
    model_choice = st.selectbox("Select Model:", ["LLaMA with RAG", "GPT-2 with RAG", "LLaMA", "GPT-2"])

    if st.button("Submit"):
        if query:
            preprocessed_input = preprocess_prompt(query)

            # Check if the preprocessed query has already been processed
            previous_chat = next((chat for chat in st.session_state.chat_history if preprocess_prompt(chat['user']) == preprocessed_input), None)

            if previous_chat:
                # Retrieve the response from chat history
                cleaned_output = previous_chat['chatbot']
                response_time = 0.00  # Since it's a cached response
                st.subheader("Response:")
                st.write(cleaned_output)
                st.write("Response Time: {:.2f} seconds (cached response)".format(response_time))

                # Set the current response in session state for feedback
                st.session_state.current_response = cleaned_output

                # Call feedback handling function
                handle_feedback(previous_chat)

            else:
                with st.spinner("Fetching response..."):
                    response, response_time = rag_response(preprocessed_input, model_choice=model_choice)
                    cleaned_output = postprocess_response(response)

                    # Store the conversation in chat history
                    chat_entry = {
                        "user": query,
                        "chatbot": cleaned_output,
                        "model": model_choice,
                        "response_time": response_time,
                        "rating": None  # Initialize rating
                    }
                    st.session_state.chat_history.append(chat_entry)

                    st.subheader("Response:")
                    st.write(cleaned_output)
                    st.write(f"Response Time: {response_time:.2f} seconds")

                    # Set the current response in session state for feedback
                    st.session_state.current_response = cleaned_output

                    # Call feedback handling function for the new response
                    handle_feedback(chat_entry)

# Conversation History Tab
with tab2:
    if st.session_state.chat_history:
        st.write("### Previous Conversations")
        for chat in st.session_state.chat_history:
            st.write(f"**User:** {chat['user']}")
            st.write(f"**Chatbot:** {chat['chatbot']}")
            st.write(f"**Model Used:** {chat['model']}")
            st.write(f"**Response Time:** {chat['response_time']:.2f} seconds")
            
            if chat.get('rating') is not None:
                st.write(f"**Rating:** {chat['rating']}")
    else:
        st.write("No conversation history yet.")

# Clear the current response display if needed
if 'current_response' in st.session_state and st.session_state.current_response is None:
    st.empty()  # This can help to clear any displayed response if required




