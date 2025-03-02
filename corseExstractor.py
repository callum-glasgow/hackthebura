import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Enable progress_apply in pandas
tqdm.pandas()

def build_index():
    # Step 1: Load your data (adjust the file path and column names as needed)
    df = pd.read_csv("/Users/callumglasgow/Downloads/year4/HACK/corseData/DATA.csv")

    # Step 2: Create a concatenated text from the most relevant fields for search.
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
    print("Sample combined text:")
    print(df['combined_text'].head())

    # Step 3: Load a pre-trained Sentence-Transformer model (BERT variant)
    model = SentenceTransformer('all-MiniLM-L12-v2')
    
    # Optionally, save the model to disk so you don't have to reload it each time.
    model_save_path = "saved_sentence_transformer_model"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Step 4: Generate embeddings for each course's combined text using a progress bar.
    df['embeddings'] = df['combined_text'].progress_apply(lambda x: model.encode(x))

    print(df[['Course Code', 'embeddings']].head())

    # -------------------------------
    # Building a Faiss index for faster similarity search
    # -------------------------------
    embedding_matrix = np.vstack(df['embeddings'].values).astype('float32')
    faiss.normalize_L2(embedding_matrix)
    d = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embedding_matrix)
    print(f"Faiss index built with {index.ntotal} vectors.")

    # Save the Faiss index to a file
    faiss_index_path = "faiss_index.index"
    faiss.write_index(index, faiss_index_path)
    print(f"Faiss index saved to: {faiss_index_path}")

    # Return the DataFrame, model, and index for later use
    return df, model, index

def search_courses(query, df, model, index, top_k=5):
    """
    Given a natural language query, compute its embedding,
    normalize it, search the Faiss index for the top_k most similar course embeddings,
    and return the matching courses with similarity scores.
    """
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]].copy()
    results['similarity'] = distances[0]
    return results[['Course Code', 'Course Name', 'similarity']]

if __name__ == "__main__":
    # Optional: Set environment variable to force single-threaded operation
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Build index (if not already built)
    df, model, index = build_index()
    
    # Now perform a search query.
    query = input("Enter your search query: ")
    results = search_courses(query, df, model, index, top_k=5)
    print(f"\nTop matching courses for query '{query}':")
    print(results)