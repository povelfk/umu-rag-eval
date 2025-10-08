import re
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def filter_synthetic_questions(
    synthetic_data: List[Dict],
    min_question_length: int = 5,
    max_question_length: int = 50,
    similarity_threshold: float = 0.95,
    remove_duplicates: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filters a dataset of synthetic questions based on length and duplicate detection.
    
    Returns two lists:
      - accepted_data: Questions that passed all filters.
      - filtered_out_data: Questions that were filtered out, with an added key "filtered_reason"
        explaining why the question was removed.

    Parameters:
      - synthetic_data: List of dictionaries, each containing a "synthetic_question" key and optionally "synthetic_embedding".
      - min_question_length: Minimum allowed question length (in words).
      - max_question_length: Maximum allowed question length (in words).
      - similarity_threshold: Cosine similarity threshold for duplicate detection (default 0.95).
      - remove_duplicates: Whether to check for duplicates using embeddings (default True).

    Returns:
      A tuple with:
         (accepted_data, filtered_out_data)
    """
    accepted_data = []
    filtered_out_data = []

    # First pass: Filter by length
    print("Filtering by length...")
    for item in tqdm(synthetic_data, desc="Length filter"):
        # Retrieve and clean the synthetic question.
        q = item.get("synthetic_question", "").strip()
        
        # Check if question is empty.
        if not q:
            item["filtered_reason"] = "Empty question"
            filtered_out_data.append(item)
            continue

        # Check length constraints (word count).
        word_count = len(q.split())
        if word_count < min_question_length:
            item["filtered_reason"] = f"Question too short: {word_count} words (min {min_question_length})"
            filtered_out_data.append(item)
            continue

        if word_count > max_question_length:
            item["filtered_reason"] = f"Question too long: {word_count} words (max {max_question_length})"
            filtered_out_data.append(item)
            continue

        accepted_data.append(item)

    # Second pass: Check for duplicates using cosine similarity
    if remove_duplicates and len(accepted_data) > 0:
        # Check if embeddings are available
        has_embeddings = "synthetic_question_embedding" in accepted_data[0]
        
        if has_embeddings:
            print("\nChecking for duplicates using embeddings...")
            
            # Extract embeddings
            embeddings = np.vstack([np.array(item["synthetic_question_embedding"]) for item in accepted_data])
            
            # Compute pairwise similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # Track indices to remove
            indices_to_remove = set()
            duplicate_pairs = []
            
            for i in range(len(similarity_matrix)):
                if i in indices_to_remove:
                    continue
                for j in range(i+1, len(similarity_matrix)):
                    if j in indices_to_remove:
                        continue
                    if similarity_matrix[i][j] >= similarity_threshold:
                        duplicate_pairs.append((i, j, similarity_matrix[i][j]))
                        indices_to_remove.add(j)  # Keep first occurrence, remove subsequent
            
            # Report findings
            if duplicate_pairs:
                print(f"\n🔍 Found {len(duplicate_pairs)} duplicate pairs (similarity >= {similarity_threshold})")
                print(f"   Removing {len(indices_to_remove)} duplicate questions...")
                
                # Show some examples
                exact_duplicates = [p for p in duplicate_pairs if p[2] == 1.0]
                near_duplicates = [p for p in duplicate_pairs if similarity_threshold <= p[2] < 1.0]
                
                if exact_duplicates:
                    print(f"   • Exact duplicates (similarity = 1.0): {len(exact_duplicates)}")
                    for i, j, sim in exact_duplicates[:2]:
                        print(f"     - \"{accepted_data[i]['synthetic_question'][:60]}...\"")
                
                if near_duplicates:
                    print(f"   • Near-duplicates ({similarity_threshold} ≤ similarity < 1.0): {len(near_duplicates)}")
                
                # Move duplicates to filtered_out_data
                final_accepted = []
                for idx, item in enumerate(accepted_data):
                    if idx in indices_to_remove:
                        item["filtered_reason"] = f"Duplicate (similarity >= {similarity_threshold})"
                        filtered_out_data.append(item)
                    else:
                        final_accepted.append(item)
                
                accepted_data = final_accepted
            else:
                print(f"✅ No duplicates found (threshold: {similarity_threshold})")
        else:
            print("⚠️ Embeddings not found. Skipping duplicate detection.")

    print(f"\n📊 Final counts:")
    print(f"   Accepted: {len(accepted_data)}")
    print(f"   Filtered out: {len(filtered_out_data)}")

    return accepted_data, filtered_out_data