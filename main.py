from utils.retriever import retrieve, rerank_cross_encoder
from utils.prompt_builder import build_gemini_prompt
from utils.retriever import print_retrieved_oneliners
from utils.gemini_client import call_gemini, get_truncated_gemini_answer
def main():
    sample_queries = [
        "What is keratoderma with woolly hair?",
        "Is keratoderma with woolly hair inherited?",
        "What are the treatments for keratoderma with woolly hair?"
    ]
    print("\nSample queries:")
    for i, q in enumerate(sample_queries, 1):
        print(f"  {i}. {q}")
    try:
        query = input("\nEnter your medical question (or press Enter to use sample 1): ").strip()
        if not query:
            query = sample_queries[0]
    except EOFError:
        print("No input detected. Using default sample query 1.")
        query = sample_queries[0]
    top_k = 10

    # Step 1: Retrieve candidates
    results = retrieve(query, k=top_k)
    print_retrieved_oneliners(results, max_items=3, maxlen=80) # Print retrieved docs as one-liners

    # Step 2: Rerank candidates
    reranked = rerank_cross_encoder(query, results)

    # Step 3: Build Gemini prompt
    gemini_prompt = build_gemini_prompt(query, reranked, max_contexts=3)

    # Step 4: Call Gemini and stream the answer
    print("\n--- Gemini LLM Answer (truncated) ---\n")
    answer = call_gemini(gemini_prompt, stream=True)
    truncated_answer = get_truncated_gemini_answer(answer, max_words=40)
    print(truncated_answer)

    # Step 5: Print sources
    print("\nSources:")
    for i, item in enumerate(reranked[:3], 1):
        url = item.get("url", "N/A")
        title = item.get("question", "")[:60]
        if len(item.get("question", "")) > 60:
            title += "..."
        print(f"{i}. {title}\n   {url}")

if __name__ == "__main__":
    main()
