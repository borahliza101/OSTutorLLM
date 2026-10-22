# OSTutorLLM

An Operating Systems tutor that combines retrieval over course material (RAG) with a fine-tuned language model. 

## What it does
- Reads 24 course files (OSTEP, slides, notes) and splits them into 2,243 chunks.
- Retrieves the three most relevant chunks for a student's question and shows the sources.
- Answers with a language model. The fine-tuned model (Qwen3-4B, QLoRA) answers in a teacher style: a direct answer, an example, and a "Quick check" question.
- We compared five configurations on 15 held-out synchronization questions (base, base + teacher prompt, fine-tuned, RAG + fine-tuned, RAG + base + prompt). Details are in the report.

## Repository layout
- `src/ingestion`: text extraction, cleaning and chunking
- `src/retrieval`: embeddings (BAAI/bge-small-en-v1.5), ChromaDB, retrieval tests
- `src/generation`: the RAG answer pipeline
- `src/fine_tuning`: dataset preparation and checks
- `src/evaluation`: dataset checks and evaluation scripts
- `data/fine_tuning`: synchronization Q&A datasets (round 1 textbook style, round 2 teacher style) and the test set
- `data/evaluation`: RAG evaluation questions
- `notebooks`: Colab notebook for QLoRA fine-tuning and the comparison
- `app.py`: Streamlit dashboard (laptop, small base model, shows retrieved sources)

## How to run
1. `pip install -r requirements.txt`
2. Build the knowledge base: run the scripts in `src/ingestion` (extract, clean, chunk), then `src/retrieval/create_embeddings.py`.
3. Start the llama.cpp server on port 8080 with the Qwen2.5-1.5B-Instruct model.
4. `streamlit run app.py`

The fine-tuning and the fine-tuned comparison run in Google Colab (free T4 GPU) using the notebook in `notebooks`.

## Team
- Liza Borah (24BKT0093): ingestion, retrieval, RAG pipeline, Streamlit dashboard, evaluation setup
- Nisha Ahlawat (24BKT0109): QLoRA fine-tuning (notebook and training)
