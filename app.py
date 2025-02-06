import os
import json
import faiss
import numpy as np
import requests
import time
import re
import random
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load corpus from an external JSON file (corpus.json)
try:
    with open('corpus.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)
    print("✅ Corpus loaded successfully.")
except Exception as e:
    print("❌ Error loading corpus file:", e)
    docs = []

# Create FAISS index for embeddings
dimension = 384  # Dimension of the embeddings (matches the model's output)
index = faiss.IndexFlatL2(dimension)

if docs:
    doc_embeddings = embedding_model.encode(docs)
    index.add(np.array(doc_embeddings, dtype=np.float32))
else:
    print("⚠️ No documents loaded in corpus. The index will remain empty.")

# Load the Hugging Face API key from an environment variable
api_key = os.environ.get("HF_API_KEY")
if not api_key:
    raise Exception("HF_API_KEY environment variable not set!")

def retrieve_relevant_text(query, top_k=3):
    """
    Retrieve the top k most relevant documents from the corpus.
    """
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [docs[indices[0][i]] for i in range(top_k)]

def choose_response_style(query):
    """
    Decide whether the answer should be concise or detailed based on keywords.
    """
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["detailed", "explain in detail", "elaborate", "in detail"]):
        return "detailed"
    return "concise"

def generate_prompt(query, retrieved_texts):
    """
    Generate a prompt by randomly selecting a template.
    The templates explicitly instruct the model to base its answer solely on the provided context.
    If the requested information is missing, it should state so.
    The answer must be in first person, as if Khalid himself is speaking.
    """
    response_style = choose_response_style(query)
    
    # Concise templates
    concise_templates = [
        (
            "You are an AI assistant with in-depth knowledge about Khalid Mehtab Khan. Below is the background information:\n\n"
            "---\n{retrieved_texts}\n---\n\n"
            "Answer the query: \"{query}\" using ONLY the information provided above. Do not add or invent any details. "
            "If the requested information is not available, explicitly state that it is not in the records. "
            "Answer as if you are Khalid, speaking in first-person with a personable, preppy tone. "
            "Output only your final answer on a new line starting with 'Final Answer:'.\n\n"
            "Final Answer:\n"
        ),
        (
            "You are an AI assistant who knows everything about Khalid Mehtab Khan. Here are the key details:\n\n"
            "---\n{retrieved_texts}\n---\n\n"
            "Answer the query: \"{query}\" solely using the above context. Do not include any information not provided. "
            "If a specific detail is missing, state that the information is unavailable. "
            "Your answer should be concise, engaging, and in a friendly, preppy tone in first-person, as if you are Khalid. "
            "Output only your final answer on a new line beginning with 'Final Answer:'.\n\n"
            "Final Answer:\n"
        )
    ]
    
    # Detailed templates
    detailed_templates = [
        (
            "You are an AI assistant with deep insights into Khalid Mehtab Khan’s background. Using the detailed information provided below:\n\n"
            "---\n{retrieved_texts}\n---\n\n"
            "Answer the query: \"{query}\" with a detailed response based solely on the above context. "
            "Do not add any external details; if the information is missing, state that it is unavailable. "
            "Answer in first-person with a personable, preppy tone, including specific examples where possible. "
            "Output only your final answer on a new line starting with 'Final Answer:'.\n\n"
            "Final Answer:\n"
        ),
        (
            "You are an AI assistant. Based solely on the detailed background information below about Khalid Mehtab Khan:\n\n"
            "---\n{retrieved_texts}\n---\n\n"
            "Answer the query: \"{query}\" with a detailed, personalized response. Use only the provided context and do not invent any details. "
            "If the requested detail is missing, state so explicitly. "
            "Your answer should be engaging, precise, and in a friendly tone in first-person, as if you are Khalid. "
            "Output only your final answer on a new line beginning with 'Final Answer:'.\n\n"
            "Final Answer:\n"
        )
    ]
    
    if response_style == "detailed":
        template = random.choice(detailed_templates)
    else:
        template = random.choice(concise_templates)
        
    return template.format(retrieved_texts=retrieved_texts, query=query)

def generate_response(prompt):
    """
    Call the Hugging Face LLM API with a retry mechanism.
    """
    models = [
        "HuggingFaceH4/zephyr-7b-beta"  # Additional models can be added if needed.
    ]
    url_template = "https://api-inference.huggingface.co/models/{}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for hf_model in models:
        url = url_template.format(hf_model)
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.5,
                "max_new_tokens": 300
            }
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                response_data = response.json()
                if isinstance(response_data, list) and len(response_data) > 0:
                    return response_data[0].get("generated_text", "No text generated.")
        except Exception:
            pass  # If an error occurs, try the next model
        time.sleep(5)
    
    return "Error: All LLM APIs are currently unavailable. Please try again later."

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Please provide a query in the request body."}), 400
    query = data["query"]
    
    # Retrieve relevant documents from the corpus
    retrieved_docs = retrieve_relevant_text(query)
    retrieved_texts = "\n".join(retrieved_docs)
    
    # Generate a prompt using the query and context
    prompt = generate_prompt(query, retrieved_texts)
    
    # Get the generated response
    full_response = generate_response(prompt)
    
    # Extract only the final answer if present
    if "Final Answer:" in full_response:
        final_answer = full_response.split("Final Answer:")[-1].strip()
    else:
        final_answer = full_response.strip()
    
    return jsonify({"answer": final_answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
