import yaml

from rag.retriever import retriever

from langchain_ollama import OllamaLLM

# ============================================================
# LOAD LLM CONFIGURATION
# ============================================================

with open("llm/llm_config.yaml", "r") as file:
    llm_config = yaml.safe_load(file)

# ============================================================
# LOAD PROMPT TEMPLATE
# ============================================================

with open("rag/prompts/prompt_templates.yaml", "r") as file:
    prompt_data = yaml.safe_load(file)

template = prompt_data["template"]

# ============================================================
# INITIALIZE OLLAMA LLM
# ============================================================

llm = OllamaLLM(
    model=llm_config["model"],
    temperature=llm_config["temperature"]
)

# ============================================================
# RAG RESPONSE FUNCTION
# ============================================================

def generate_response(query: str):

    # --------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    docs = retriever.get_relevant_documents(query)

    # --------------------------------------------------------
    # BUILD CONTEXT FROM RETRIEVED DOCS
    # --------------------------------------------------------

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # --------------------------------------------------------
    # FORMAT FINAL PROMPT
    # --------------------------------------------------------

    final_prompt = template.format(
        context=context,
        query=query
    )

    # --------------------------------------------------------
    # GENERATE RESPONSE FROM LLM
    # --------------------------------------------------------

    response = llm.invoke(final_prompt)

    return response


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    user_query = "Suggest a modern sofa under 40000"

    result = generate_response(user_query)

    print("\nAI RESPONSE:\n")
    print(result)