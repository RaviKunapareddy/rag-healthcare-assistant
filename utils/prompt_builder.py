def build_gemini_prompt(query, top_contexts, max_contexts=3, instruction=None):
    """
    Dynamically constructs a prompt for Gemini LLM using the top reranked Q&A pairs as context.
    Args:
        query (str): The user's question.
        top_contexts (list): List of dicts with 'question' and 'answer' keys.
        max_contexts (int): Number of context Q&A pairs to include.
        instruction (str, optional): Custom instruction for LLM. If None, uses default.
    Returns:
        str: The constructed prompt.
    """
    context_strs = []
    for i, item in enumerate(top_contexts[:max_contexts], 1):
        context_strs.append(f"Q{i}: {item['question']}\nA{i}: {item['answer']}")
    context_block = "\n\n".join(context_strs)
    base_instruction = (
        "Using only the above context, answer the user's question as accurately and concisely as possible. "
        "If the answer is not found in the context, reply: 'Not found in context.'"
    )
    prompt = (
        f"Context:\n{context_block}\n\n"
        f"{instruction or base_instruction}\n"
        f"User Question: {query}\n"
    )
    return prompt
