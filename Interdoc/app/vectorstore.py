import re

# Lightweight local storage for uploaded document chunks.
# This keeps the app fully functional without requiring a heavy embedding model.
chunks_store: list[dict] = []


def add_chunks(doc_id: str, chunks: list[str]):
    for index, chunk in enumerate(chunks):
        chunks_store.append({
            "id": f"{doc_id} {index}",
            "source": doc_id,
            "chunk_index": index,
            "text": chunk,
        })
    return len(chunks)


def search_chunks(query: str, top_k: int = 4):
    normalized_query = set(re.findall(r"\w+", query.lower()))
    if not normalized_query:
        return {"documents": [[]], "metadatas": [[]]}

    scored = []
    for item in chunks_store:
        text = item["text"].lower()
        tokens = set(re.findall(r"\w+", text))
        overlap = len(normalized_query & tokens)
        if overlap:
            scored.append((overlap, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_items = [item for _, item in scored[:top_k]]

    documents = [[item["text"] for item in top_items]]
    metadatas = [[{"source": item["source"], "chunk_index": item["chunk_index"]} for item in top_items]]
    return {"documents": documents, "metadatas": metadatas}