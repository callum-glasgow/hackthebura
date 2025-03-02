import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Enable progress_apply in pandas
tqdm.pandas()

def build_and_save_indices():
    # --- Configuration ---
    CSV_PATH = "/Users/callumglasgow/Downloads/year4/HACK/corseData/DATA.csv"
    MODEL_PATH = "saved_sentence_transformer_model2"
    indices_dir = "faiss_indices_by_year"
    
    # --- Load Course Data ---
    df = pd.read_csv(CSV_PATH)
    
    # Ensure 'Programme Year' is treated as a trimmed string.
    df['Programme Year'] = df['Programme Year'].astype(str).str.strip()
    
    # Create a concatenated text field from relevant columns.
    df['combined_text'] = (
        "Course Name: " + df['Course Name'].astype(str) + " | " +
        "Programme Name: " + df['Programme Name'].astype(str) + " | " +
        "Programme Subject: " + df['Programme Subject'].astype(str) + " | " +
        "Programme School Name: " + df['Programme School Name'].astype(str) + " | " +
        "Programme Year: " + df['Programme Year'] + " | " +
        "Study Level: " + df['Study Level'].astype(str) + " | " +
        "Compulsory/Optional: " + df['Compulsory/Optional'].astype(str) + " | " +
        "Core/Elective: " + df['Core/Elective'].astype(str) + " | " +
        "Course Subject: " + df['Course Subject'].astype(str)
    )
    print("Sample combined text:")
    print(df['combined_text'].head())
    
    # --- Load Sentence-Transformer Model ---
    model = SentenceTransformer('all-MiniLM-L12-v2')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    # --- Generate Embeddings ---
    print("Generating embeddings for each course...")
    df['embeddings'] = df['combined_text'].progress_apply(lambda x: model.encode(x))
    print(df[['Course Code', 'embeddings']].head())
    
    # --- Create Directory for Per-Year Faiss Indices ---
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir)
    
    # Get unique Programme Years
    unique_years = df['Programme Year'].unique()
    print("Unique Programme Years found:", unique_years)
    
    # --- Build and Save a Faiss Index for Each Programme Year ---
    for year in unique_years:
        df_year = df[df['Programme Year'] == year].copy()
        if df_year.empty:
            continue
        
        # Build embedding matrix for current year.
        embedding_matrix = np.vstack(df_year['embeddings'].values).astype('float32')
        faiss.normalize_L2(embedding_matrix)
        d = embedding_matrix.shape[1]
        index_year = faiss.IndexFlatIP(d)
        index_year.add(embedding_matrix)
        print(f"Faiss index for Programme Year {year} built with {index_year.ntotal} vectors.")
        
        # Save the index to a file named "faiss_index_<year>.index"
        index_path = os.path.join(indices_dir, f"faiss_index_{year}.index")
        faiss.write_index(index_year, index_path)
        print(f"Faiss index for Programme Year {year} saved to: {index_path}")
    
    # Return the DataFrame and model for later use if needed.
    return df, model

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    df, model = build_and_save_indices()