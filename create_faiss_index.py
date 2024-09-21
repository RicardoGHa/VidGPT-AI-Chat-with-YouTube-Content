import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

# Load Hugging Face model and tokenizer
def load_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Chunk the transcript into smaller pieces
def chunk_text(text, max_chunk_size=512):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Generate embeddings using a transformer model
def generate_embeddings(text_chunks, tokenizer, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for chunk in text_chunks:
            inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())  # Mean pooling
    return np.array(embeddings)

# Create a Faiss index for retrieval
def create_faiss_index(embeddings, video_id):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{video_id}_faiss.index")
    return index

# Extract transcript from YouTube video
def extract_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([entry['text'] for entry in transcript])
    with open(f"{video_id}_transcript.txt", 'w') as f:
        f.write(transcript_text)
    return transcript_text

if __name__ == "__main__":
    video_id = input("Enter YouTube video ID: ")

    # Step 1: Extract transcript
    transcript_text = extract_transcript(video_id)

    # Step 2: Load the model and tokenizer
    tokenizer, model = load_model()

    # Step 3: Chunk the transcript
    chunks = chunk_text(transcript_text)

    # Step 4: Generate embeddings
    embeddings = generate_embeddings(chunks, tokenizer, model)

    # Step 5: Create and save Faiss index
    index = create_faiss_index(embeddings, video_id)

    print(f"FAISS index created and saved as {video_id}_faiss.index.")
