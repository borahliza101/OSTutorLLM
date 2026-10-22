import chromadb
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


# ------------------------------
# Settings (same values as rag_pipeline.py)
# ------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "os_tutor"
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
TOP_K = 3

SYSTEM_PROMPT = """
You are an Operating Systems tutor.

Answer the student's question using the provided
Operating Systems study material.

Use the retrieved material as your primary source.

If the material does not contain enough information
to answer the question, say so instead of inventing
facts.

Explain concepts clearly and at an appropriate level
for a university student.
"""


# ------------------------------
# Load the embedding model and the database ONCE.
# Streamlit re-runs this whole file on every click,
# so without caching it would reload the model each time.
# ------------------------------
@st.cache_resource
def load_resources():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    return model, collection


embedding_model, collection = load_resources()


# ------------------------------
# Page
# ------------------------------
st.title("OSTutorLLM")
st.caption("Operating Systems tutor: RAG baseline (no fine-tuning yet)")
st.write("Chunks in knowledge base:", collection.count())

question = st.text_input("Ask an Operating Systems question:")

if st.button("Ask") and question.strip():

    # Step 1: turn the question into an embedding
    question_embedding = embedding_model.encode(question).tolist()

    # Step 2: find the TOP_K closest chunks in ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Step 3: put the chunks and the question into a prompt
    context = "\n\n".join(documents)

    user_prompt = f"""
Here is the retrieved Operating Systems study material:

--------------------
{context}
--------------------

Student question:

{question}

Answer the question clearly using the study material.
"""

    # Step 4: send the prompt to the local LLM server
    try:
        with st.spinner("Generating answer..."):
            response = requests.post(
                LLM_URL,
                json={
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.2
                },
                timeout=180
            )
            response.raise_for_status()
    except requests.exceptions.RequestException as error:
        st.error("Could not reach the LLM server. Is it running?")
        st.write(error)
        st.stop()

    choice = response.json()["choices"][0]

    # Step 5: show the answer
    st.subheader("Answer")
    st.write(choice["message"]["content"])

    if choice["finish_reason"] == "length":
        st.warning("This answer was cut off because it hit the token limit.")

    # Step 6: show what was retrieved, so the pipeline is visible
    st.subheader("Retrieved sources")
    for i in range(len(documents)):
        title = f"{i + 1}. {metadatas[i]['source']} (distance {distances[i]:.3f})"
        with st.expander(title):
            st.text(documents[i])