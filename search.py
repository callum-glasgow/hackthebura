import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---
DEFAULT_FAISS_INDEX_PATH = "faiss_index.index"
YEAR_INDEX_DIR = "/Users/callumglasgow/Downloads/year4/HACK/faiss_indices_by_year"
MODEL_PATH = "saved_sentence_transformer_model"
CSV_PATH = "/Users/callumglasgow/Downloads/year4/HACK/corseData/DATA.csv"

# --- Load Saved Model ---
model = SentenceTransformer(MODEL_PATH)
print("Loaded SentenceTransformer model from:", MODEL_PATH)

# --- Ask user if they want to restrict search to a specific Programme Year ---
year_input = input("Enter a specific Programme Year (1-7) to restrict search, or press Enter to use the full index: ").strip()
year_specific = False

if year_input and year_input in [str(i) for i in range(1, 8)]:
    index_path = os.path.join(YEAR_INDEX_DIR, f"faiss_index_{year_input}.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Faiss index file not found at {index_path}.")
    index = faiss.read_index(index_path)
    print(f"Loaded Faiss index for Programme Year {year_input} with {index.ntotal} vectors.")
    year_specific = True
else:
    if not os.path.exists(DEFAULT_FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Faiss index file not found at {DEFAULT_FAISS_INDEX_PATH}.")
    index = faiss.read_index(DEFAULT_FAISS_INDEX_PATH)
    print(f"Loaded full Faiss index with {index.ntotal} vectors.")

# --- Load Course Data ---
df = pd.read_csv(CSV_PATH)

# Recreate the combined text exactly as it was used for indexing.
df['combined_text'] = (
    "Course Name: " + df['Course Name'].astype(str) + " | " +
    "Programme Name: " + df['Programme Name'].astype(str) + " | " +
    "Programme Subject: " + df['Programme Subject'].astype(str) + " | " +
    "Programme School Name: " + df['Programme School Name'].astype(str) + " | " +
    "Programme Year: " + df['Programme Year'].astype(str) + " | " +
    "Study Level: " + df['Study Level'].astype(str) + " | " +
    "Compulsory/Optional: " + df['Compulsory/Optional'].astype(str) + " | " +
    "Core/Elective: " + df['Core/Elective'].astype(str) + " | " +
    "Course Subject: " + df['Course Subject'].astype(str)
)
print("Course data loaded.")

# If a specific year is selected, filter the DataFrame accordingly.
if year_specific:
    df_index = df[df['Programme Year'].astype(str) == year_input].copy()
    print(f"Filtered course data for Programme Year {year_input}: {len(df_index)} courses found.")
else:
    df_index = df

def search_courses(query, df_index, index, unique_count=10, initial_top_k=10):
    """
    Given a natural language query, this function:
      1. Encodes the query and normalizes the embedding.
      2. Uses the loaded Faiss index to retrieve the top_k nearest courses.
      3. Drops duplicate courses (based on 'Course Name') and dynamically increases the number
         of nearest neighbors searched until at least 'unique_count' unique results are found,
         or until top_k equals the maximum available results.
    Returns a DataFrame with Course Name, Programme Year, Course Subject, and similarity score.
    """
    # Encode and normalize the query embedding.
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    top_k = initial_top_k
    max_results = index.ntotal
    
    while top_k <= max_results:
        distances, indices = index.search(query_embedding, top_k)
        results = df_index.iloc[indices[0]].copy()
        results['similarity'] = distances[0]
        # Drop duplicate course names.
        unique_results = results.drop_duplicates(subset='Course Name')
        
        if len(unique_results) >= unique_count or top_k == max_results:
            return unique_results.head(unique_count)
        else:
            top_k = min(top_k * 2, max_results)
    
    return unique_results

# --- Main Execution ---
if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = search_courses(query, df_index, index, unique_count=10, initial_top_k=10)
    
    if results.empty:
        print(f"No results found for query '{query}'.")
    else:
        print(f"\nTop matching courses for query '{query}':")
        print(results[['Course Name', 'Programme Year', 'Course Subject', 'similarity']])