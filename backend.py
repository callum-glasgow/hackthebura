from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)
# Allow requests from specified origins
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000", "http://localhost", "http://127.0.0.1"
]}})

# --- Configuration ---
DEFAULT_FAISS_INDEX_PATH = "faiss_index.index"
YEAR_INDEX_DIR = "/Users/callumglasgow/Downloads/year4/HACK/faiss_indices_by_year"
MODEL_PATH = "saved_sentence_transformer_model"
CSV_PATH = "/Users/callumglasgow/Downloads/year4/HACK/corseData/DATA.csv"
SYNTHETIC_CSV_PATH = "/Users/callumglasgow/Downloads/year4/HACK/corseData/courses_synthetic.csv"  # Additional CSV

# New paths for stats data
ASSESSMENTS_SYNTHETIC_CSV_PATH = "/Users/callumglasgow/Downloads/year4/HACK/corseData/assessments_synthetic.csv"
COURSES_SYNTHETIC_CSV_PATH = "/Users/callumglasgow/Downloads/year4/HACK/corseData/corseData/courses_synthetic.csv"  # If different from SYNTHETIC_CSV_PATH

# --- Define headers (order) for the returned data ---
# IMPORTANT: Include "Course Code" so that it can be used for synthetic lookup.
data_headers = [
    "Compulsory/Optional",
    "Core/Elective",
    "Course Code",         
    "Course Name",
    "Course School Name",
    "Course Subject",
    "Programme Year",
    "Study Level",
    "similarity"
]

synthetic_headers = [
    "average_mark",
    "credits",
    "median",
    "mode",
    "number_of_students",
    "pass_rate",
    "semester",
    "std_dev",
    "year"
]

# --- Load Saved Model ---
model = SentenceTransformer(MODEL_PATH)
print("Loaded SentenceTransformer model from:", MODEL_PATH)

# --- Load Primary Course Data (DATA.csv) ---
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

# --- Provided Function for Synthetic Data ---
def filter_courses_by_code(course_code):
    # Read the CSV file for courses synthetic data
    df_synth = pd.read_csv(SYNTHETIC_CSV_PATH)
    # Filter rows where 'Course Code' equals the specified course_code and 'year' equals 2024.
    filtered_df = df_synth[(df_synth['Course Code'] == course_code) & (df_synth['year'] == 2024)]
    return filtered_df

def load_index_and_filter(year_input):
    """
    Load the appropriate Faiss index and filter the DataFrame based on year_input.
    If a valid year (string "1" through "7") is provided, load that year's index.
    Otherwise, load the full index and DataFrame.
    """
    if year_input and year_input in [str(i) for i in range(1, 8)]:
        index_path = os.path.join(YEAR_INDEX_DIR, f"faiss_index_{year_input}.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Faiss index file not found at {index_path}.")
        index = faiss.read_index(index_path)
        df_index = df[df['Programme Year'].astype(str) == year_input].copy()
        print(f"Loaded Faiss index for Programme Year {year_input} with {index.ntotal} vectors.")
    else:
        if not os.path.exists(DEFAULT_FAISS_INDEX_PATH):
            raise FileNotFoundError(f"Faiss index file not found at {DEFAULT_FAISS_INDEX_PATH}.")
        index = faiss.read_index(DEFAULT_FAISS_INDEX_PATH)
        df_index = df.copy()
        print(f"Loaded full Faiss index with {index.ntotal} vectors.")
    return index, df_index

def search_courses(query, df_index, index, unique_count=10, initial_top_k=10):
    """
    Encode the query, normalize the embedding, and use the Faiss index to retrieve
    the top matching courses. Duplicate courses (by 'Course Name') are dropped.
    """
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    top_k = initial_top_k
    max_results = index.ntotal
    
    while top_k <= max_results:
        distances, indices = index.search(query_embedding, top_k)
        results = df_index.iloc[indices[0]].copy()
        results['similarity'] = distances[0]
        unique_results = results.drop_duplicates(subset='Course Name')
        if len(unique_results) >= unique_count or top_k == max_results:
            return unique_results.head(unique_count)
        else:
            top_k = min(top_k * 2, max_results)
    return unique_results

def get_stats_for_course(course_code):
    """
    For each year from 2020 to 2024, load the assessments and courses synthetic CSVs,
    filter by the given course code, and return a dict keyed by year.
    """
    stats = {}
    for year in range(2020, 2025):
        year_str = str(year)
        # Load assessments synthetic data
        try:
            df_assess = pd.read_csv(ASSESSMENTS_SYNTHETIC_CSV_PATH)
            df_assess_year = df_assess[(df_assess['Course Code'] == course_code) & (df_assess['year'] == year)]
        except Exception as e:
            df_assess_year = pd.DataFrame()
            print(f"Error loading assessments for year {year}: {e}")

        # Load courses synthetic data
        try:
            df_courses = pd.read_csv(SYNTHETIC_CSV_PATH)
            df_courses_year = df_courses[(df_courses['Course Code'] == course_code) & (df_courses['year'] == year)]
        except Exception as e:
            df_courses_year = pd.DataFrame()
            print(f"Error loading courses for year {year}: {e}")
            
        stats[year_str] = {
            "assessments": df_assess_year.to_dict(orient="records"),
            "courses": df_courses_year.to_dict(orient="records")
        }
    return stats

@app.route('/', methods=['GET'])
def home():
    """Simple endpoint to confirm the server is running."""
    return jsonify({"message": "Flask search API is running."})

@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    Expects a JSON payload with:
      - query: the search string (for query requests).
      - year: (optional) a specific Programme Year (string "1"–"7").
      - request_type: "query" or "stats". If "stats", then also expect "Course Code".
    """
    req_data = request.get_json()
    request_type = req_data.get('request_type', 'query').strip().lower()
    
    # If this is a stats request, process accordingly.
    if request_type == "stats":
        course_code = req_data.get("Course Code", "").strip()
        if not course_code:
            return jsonify({"error": "No Course Code provided for stats request"}), 400
        
        stats = get_stats_for_course(course_code)
        return jsonify({"stats": stats})
    
    # Otherwise, handle as a normal query search.
    query = req_data.get('query', '').strip()
    year_input = req_data.get('year', '').strip()
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        index, df_index = load_index_and_filter(year_input)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    results_df = search_courses(query, df_index, index, unique_count=10, initial_top_k=10)
    
    if results_df.empty:
        return jsonify({"results": [], "message": f"No results found for query '{query}'."})
    
    # Ensure we return columns in the original order.
    base_cols = list(df.columns) + ['similarity']
    results_df = results_df[base_cols]
    
    final_results = []
    for _, row in results_df.iterrows():
        base_data = row.to_dict()
        # Reorder the base data according to data_headers.
        ordered_base_data = {header: base_data.get(header) for header in data_headers}
        
        course_code = ordered_base_data.get("Course Code")
        synth_df = filter_courses_by_code(course_code)
        synthetic_data = synth_df.to_dict(orient="records") if not synth_df.empty else []
        
        # Merge synthetic data if available (take only the first record)
        merged_result = dict(ordered_base_data)
        if synthetic_data:
            ordered_synth = {header: synthetic_data[0].get(header) for header in synthetic_headers}
            merged_result.update(ordered_synth)
        
        final_results.append(merged_result)
    
    return jsonify({"results": final_results})

if __name__ == "__main__":
    app.run(port=8000, debug=True)