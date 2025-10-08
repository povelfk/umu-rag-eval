from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from azure.search.documents import SearchClient


def get_all_chunks(index, credential, search_endpoint, k=100000):
    try:
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index,
            credential=credential,
            include_total_count=True
        )
        
        results = search_client.search(
            search_text="*",
            top=k
        )
        data = []

        pages = results.by_page()
        for page in pages:
            results = list(page)
            for result in results:
                data.append(
                    {
                        "chunk_id": result["chunk_id"], # change to id
                        "title": result["title"],
                        "chunk": result["chunk"],
                        "chunk_embedding": result["text_vector"]
                    }
                )
            return data
    except Exception as e:
        print(e)
        return None


def get_chunk(index, chunk_id, search_endpoint, credential, fields=None):
    from azure.search.documents import SearchClient
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index,
        credential=credential
    )
    if fields:
        document = search_client.get_document(
            key=chunk_id,
            selected_fields=fields
        )
    else:
        document = search_client.get_document(key=chunk_id)

    return document


def extract_page_number(chunk_id):
    """
    Extract page number from chunk_id.
    Pattern: ...pages_{number}
    Returns: int (page number) or 0 if not found
    """
    match = re.search(r'pages_(\d+)$', chunk_id)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no page number found


def build_document_map(chunks, extract_pn=False):
    """
    Build a map of document title -> list of chunks (optionally sorted by page).

    Iden va att skapa en fråga som inte kunde besvaras utifrån hela dokumentet ist för enbart en chunk
    
    """
    doc_map = defaultdict(list)
    
    print("🔍 Building document-to-chunks map...")
    for chunk in tqdm(chunks, desc="Mapping chunks to documents"):
        title = chunk.get("title", chunk["chunk_id"]) # CRTICAL ASSUMPTION!!! Title is the DOCUMENT TITLE, not the CHUNK TITLE
        
        chunk_data = {
            "chunk_id": chunk["chunk_id"],
            "chunk": chunk["chunk"],
            "title": title
        }
        
        if extract_pn:
            chunk_data["page_number"] = extract_page_number(chunk["chunk_id"])
        
        doc_map[title].append(chunk_data)
    
    if extract_pn:
        print("📑 Sorting chunks by page number...")
        for title in doc_map:
            doc_map[title].sort(key=lambda x: x.get("page_number", 0))
    
    print(f"✅ Found {len(doc_map)} unique documents")
    print(f"   Average chunks per document: {sum(len(v) for v in doc_map.values()) / len(doc_map):.1f}")
    
    return doc_map


def find_similar_chunks(main_chunk, all_chunks, k=5):
    """
    Find k most similar chunks to the main chunk using cosine similarity.
    
    Args:
        main_chunk: Dict with 'chunk_id' and 'chunk_embedding' keys
        all_chunks: List of dicts, each with 'chunk_id', 'chunk', and 'chunk_embedding'
        k: Number of similar chunks to return (default 5)
        
    Returns:
        List of k most similar chunks (excluding the main chunk itself)
    """
    main_chunk_id = main_chunk['chunk_id']
    main_embedding = np.array(main_chunk['chunk_embedding']).reshape(1, -1)
    
    # Prepare all embeddings and filter out the main chunk
    other_chunks = [c for c in all_chunks if c['chunk_id'] != main_chunk_id]
    
    if len(other_chunks) == 0:
        return []
    
    # Stack embeddings for vectorized computation
    other_embeddings = np.vstack([np.array(c['chunk_embedding']) for c in other_chunks])
    
    # Compute cosine similarities
    similarities = cosine_similarity(main_embedding, other_embeddings)[0]
    
    # Get indices of top-k most similar chunks
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    # Return the top-k similar chunks
    similar_chunks = [other_chunks[i] for i in top_k_indices]
    
    return similar_chunks