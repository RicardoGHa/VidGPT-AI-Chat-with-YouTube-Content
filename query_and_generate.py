import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


# Load Hugging Face model and tokenizer for retrieval
def load_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


# Encode the query for retrieval
def encode_query(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return query_embedding


# Retrieve relevant chunks using FAISS index
def retrieve_relevant_chunks(query_embedding, index, chunks, top_k=10):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[idx] for idx in indices[0] if idx < len(chunks)]


# Load the FAISS index from file
def load_faiss_index(index_file):
    return faiss.read_index(index_file)


# Load Hugging Face generation model (e.g., BART for summarization)
def load_generation_model(model_name="facebook/bart-large-cnn"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# Generate an answer from the retrieved chunks
def generate_answer(retrieved_chunks, question, tokenizer, model):
    # Combine all retrieved chunks
    input_text = " ".join(retrieved_chunks)

    # Use summarization model to refine and answer the question
    inputs = tokenizer(f"summarize: {input_text}", return_tensors="pt", truncation=True)
    outputs = model.generate(
        inputs.input_ids, max_length=150, num_beams=4, early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# wawMjJUCMVw streamlit run app.py https://youtu.be/JCnvVaXEh3Y?si=iSHg9WGNEIXsTKKy
