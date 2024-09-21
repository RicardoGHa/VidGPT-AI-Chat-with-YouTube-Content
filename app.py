import streamlit as st
import os
from query_and_generate import (
    load_faiss_index,
    load_model,
    encode_query,
    retrieve_relevant_chunks,
    load_generation_model,
    generate_answer,
)
from create_faiss_index import (
    extract_transcript,
    load_model,
    chunk_text,
    generate_embeddings,
    create_faiss_index,
)

st.title("VidGPT: AI Chat with YouTube Content")

# Step 1: Get YouTube video ID
youtube_link = st.text_input("Enter YouTube video link:")
if youtube_link:
    # Extract video ID
    if "youtube.com" in youtube_link:
        video_id = youtube_link.split("v=")[-1].split("&")[0]
    elif "youtu.be" in youtube_link:
        video_id = youtube_link.split("/")[-1].split("?")[0]

    transcript_file = f"{video_id}_transcript.txt"
    index_file = f"{video_id}_faiss.index"

    # Check if transcript and index files exist
    if not os.path.exists(transcript_file):
        st.write("Transcript not found, attempting to extract transcript...")
        transcript_text = extract_transcript(video_id)
        chunks = chunk_text(transcript_text)
    else:
        with open(transcript_file, "r") as f:
            chunks = f.read().splitlines()

    if not os.path.exists(index_file):
        st.write("FAISS index not found, creating index...")
        tokenizer, model = load_model()
        embeddings = generate_embeddings(chunks, tokenizer, model)
        index = create_faiss_index(embeddings, video_id)
    else:
        index = load_faiss_index(index_file)

    # Load retrieval and summarization models
    retrieval_tokenizer, retrieval_model = load_model()
    generation_tokenizer, generation_model = load_generation_model()

    # Step 2: User asks a question
    question = st.text_input("Ask a question about the video:")

    if question:
        # Encode the query
        query_embedding = encode_query(question, retrieval_tokenizer, retrieval_model)

        # Retrieve more relevant chunks (increase to top_k=10)
        relevant_chunks = retrieve_relevant_chunks(
            query_embedding, index, chunks, top_k=10
        )

        # Generate a clearer answer using a summarization model
        answer = generate_answer(
            relevant_chunks, question, generation_tokenizer, generation_model
        )

        st.write(f"Answer: {answer}")
