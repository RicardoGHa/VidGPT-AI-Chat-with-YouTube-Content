# VidGPT: AI Chat with YouTube Content

VidGPT is an AI-powered application that allows users to interact with YouTube video transcripts by asking questions and receiving concise answers. The system extracts video transcripts from YouTube, processes the transcript using a transformer model to generate embeddings, indexes the chunks with FAISS for fast retrieval, and uses a summarization model to generate clear answers to the user's questions.

## Features

- **YouTube Transcript Extraction**: Automatically fetches transcripts from YouTube videos.
- **Transformer-based Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for efficient sentence embeddings.
- **FAISS Indexing**: Fast retrieval of relevant chunks from the transcript using FAISS.
- **Summarization Model**: Uses `facebook/bart-large-cnn` to summarize and generate concise answers to user queries.
- **Streamlit Interface**: A simple user interface to ask questions about YouTube video content.

## Technologies Used

- Python 3.8+
- [PyTorch](https://pytorch.org/) - For running Hugging Face models.
- [FAISS](https://github.com/facebookresearch/faiss) - For fast similarity search.
- [Transformers](https://huggingface.co/transformers/) - Hugging Face library to load pre-trained models.
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) - To extract transcripts from YouTube.
- [Streamlit](https://streamlit.io/) - For the interactive web interface.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RicardoGHa/VidGPT-AI-Chat-with-YouTube-Content
cd VidGPT-AI-Chat-with-YouTube-Content
```
### 2. Install Dependencies
First, ensure you have Python 3.8 or higher. Then, install the required packages using pip.
```bash
pip install torch numpy faiss-cpu transformers youtube-transcript-api nltk streamlit
```
### 3. NLTK Tokenizer Setup
Run the following code to download the necessary NLTK data for tokenizing the transcript:
```python
import nltk
nltk.download('punkt')
```
### 4. Running the Application
Start the Streamlit app with the following command:
```bash
streamlit run app.py
```
The application will start, and a browser window will open with the user interface. Enter a YouTube video link, and then ask questions about the content.

### Usage
Input YouTube Video Link: Paste a YouTube video URL into the input box.
Ask a Question: After the transcript is processed, ask any question related to the video’s content.
Receive an Answer: VidGPT will retrieve the most relevant chunks of the transcript and generate a concise answer using a summarization model.

### Example
Video Link: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Question: "What is the main idea of this video?"
Answer: "The video discusses ..."

vidgpt/
│
├── app.py                         # Main Streamlit app for user interaction
├── query_and_generate.py          # Functions for querying and generating answers
├── create_faiss_index.py          # Functions for processing transcripts and creating FAISS index          
└── README.md                      # Project documentation

### Customization
Model Selection: 
 * You can replace the sentence embedding or generation models with any Hugging Face models by changing the model names in query_and_generate.py.
 * Top-k Chunks: The number of retrieved chunks can be adjusted in the function retrieve_relevant_chunks for better precision.

### FAQ

### 1. What if the transcript is unavailable?
The application depends on the YouTube Transcript API. If a transcript is unavailable (due to video restrictions), VidGPT will not be able to retrieve the content.

### 2. Can I use a different generation model?
Yes! Simply update the load_generation_model function in query_and_generate.py with the desired Hugging Face model, like t5-base for question-answering or summarization models.

### 3. How can I improve the model's answers?
You can:

Increase the top_k parameter in retrieve_relevant_chunks to retrieve more relevant chunks.
Fine-tune a model for the specific task of question-answering on transcripts.

### Acknowledgments

* Hugging Face for providing pre-trained transformer models.
* FAISS for efficient similarity search.
* YouTube Transcript API for extracting video transcripts.


### Steps:
1. Copy this content into a `README.md` file in your project folder.
2. Adjust the `git clone` link to point to your repository.
3. Optionally, add the `requirements.txt` file with your dependencies.

This document should help anyone setting up or using your project!
