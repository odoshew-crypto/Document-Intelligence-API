import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None


def build_prompt(question: str, chunks: list[str], metadatas: list[dict]):
    blocks = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), start=1):
        blocks.append(f"[Source {i}: {meta['source']}, chunk {meta['chunk_index']}]\n{chunk}")
    context = "\n\n".join(blocks)
    return (
        "Answer the question using ONLY the context below. "
        "Cite sources using [Source N]. If the answer is not in the context, "
        "say you don't have enough information - do not guess.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


def fallback_answer(question: str, chunks: list[str], metadatas: list[dict]) -> str:
    if not chunks:
        return "No documents have been uploaded yet."

    matching_chunk = next((chunk for chunk in chunks if question.lower() in chunk.lower()), chunks[0])
    snippet = matching_chunk.strip()
    if len(snippet) > 900:
        snippet = snippet[:900] + "..."
    return (
        "I grounded this answer in the most relevant uploaded passage. "
        f"The relevant content says: {snippet}"
    )


def generate_answer(question: str, search_results: dict):
    chunks = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]

    if not chunks:
        return {"answer": "No documents have been uploaded yet", "sources": []}

    try:
        if client:
            prompt = build_prompt(question, chunks, metadatas)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = response.choices[0].message.content
        else:
            raise RuntimeError("No Groq API key configured")
    except Exception:
        answer_text = fallback_answer(question, chunks, metadatas)

    sources = [{"source": m["source"], "chunk_index": m["chunk_index"]} for m in metadatas]
    return {"answer": answer_text, "sources": sources}
    