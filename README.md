
# Healthcare RAG Assistant: Retrieval, Rerank, Respond

In a world flooded with healthcare misinformation, finding trusted medical answers isn't just hard — it's risky.  
This project builds a **Retrieval-Augmented Generation (RAG)** healthcare assistant that retrieves trusted content from the **MedQuAD** dataset and generates human-like responses with **Google's Gemini LLM**, ensuring relevance, transparency, and source attribution at every step.

Built for **real-world reliability**, not just demo purposes.

---

## 🚀 Why This Project?

Most AI chatbots guess when they don't know — dangerous in healthcare.  
This system is different: it **retrieves real medical Q&A pairs**, **reranks them semantically**, and then **generates answers grounded in the retrieved evidence**.

✅ No blind hallucinations  
✅ Clear source links  
✅ Designed for production trustworthiness

It’s a real-world demonstration of how RAG pipelines can add value where trust matters most.

---

## 📚 About the Dataset

The assistant is powered by the **MedQuAD** dataset — a trusted collection of real-world medical questions and answers sourced from authoritative sites like **NIH**, **MedlinePlus**, and **Genetic and Rare Diseases Information Center (GARD)**.  
Before use, the dataset was **cleaned and preprocessed** into structured dictionaries containing:
- `question`
- `answer`
- `source`
- `filename`
- `url`

This ensures that retrieval is grounded in **validated medical information**, not random internet content.

---

## 🛠️ How It Works

Here’s what happens when you ask a question:

1. **Retrieval from FAISS Index**  
   → Your question is embedded and matched against a dense vector index built from MedQuAD Q&A pairs.

2. **Semantic Reranking**  
   → Top retrieved candidates are reranked using a CrossEncoder model for true semantic relevance.

3. **Prompt Construction**  
   → The top reranked results are packed neatly into a structured prompt format.

4. **Answer Generation with Gemini**  
   → Gemini reads the context and generates a grounded, human-like answer based on the provided evidence.

5. **Source Attribution**  
   → The final output includes links to the original medical sources — making the system trustworthy and verifiable.

---

## 📚 Technologies Used

- **Vector Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Retrieval Engine**: FAISS
- **Semantic Reranking**: CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- **LLM**: Google Gemini (`gemini-1.5-pro`)
- **Backend Glue**: Python (pandas, pickle, dotenv)

---

## 📸 Sample Outputs

- **Retrieval and Top-3 Reranked Results**  
  _Top Q&A pairs retrieved and semantically reranked based on user query._

  <img src="assets/retrieving.png" width="600"/>

- **Pipeline Overview**  
  _Internal flow showing retrieval, reranking, prompt building, and LLM answer generation._

  <img src="assets/final_pipeline.png" width="600"/>

---

## 🛤️ Internal Architecture (Code Flow)
-
    <img src="assets/architecture.png" width="600"/>


---

## 📂 Code Walkthrough

Each module has a clean, defined role:

| Stage | File | Responsibility |
|:------|:-----|:----------------|
| **Data Loading** | `data_loader.py` | Load MedQuAD Q&A into structured dictionaries |
| **Embedding & Indexing** | `embeddings.py` | Generate dense vectors and build FAISS index |
| **Retrieval** | `retriever.py` | Embed user query, retrieve top-k candidates |
| **Semantic Reranking** | `retriever.py` | Cross-encode (query, answer) for better relevance |
| **Prompt Creation** | `prompt_builder.py` | Build structured Gemini-ready prompt |
| **LLM Answering** | `gemini_client.py` | Call Gemini API, handle response |
| **Final Output** | `main.py` | Display answer and original sources |

---

## 📬 Contact Me

If you found this project useful, inspiring, or have any questions — feel free to connect!

- 🧑‍💼 **Raviteja Kunapareddy**
- 📧 Email: [ravitejakunapareddy09@gmail.com](mailto:ravitejakunapareddy09@gmail.com)
- 🌐 LinkedIn: [linkedin.com/in/ravi-kunapareddy](https://www.linkedin.com/in/ravi-kunapareddy/)
- 💼 Portfolio: [github.com/RaviKunapareddy/RaviKunapareddy](https://github.com/RaviKunapareddy/RaviKunapareddy)

*Let’s build smarter systems together — one intelligent agent at a time.* 🚀

