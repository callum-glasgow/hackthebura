import pandas as pd
import numpy as np
import random
from tqdm import tqdm  # progress bar

# --- Configuration ---
# Define the years for which to generate data
years = [2020, 2021, 2022, 2023,2024]

# Define the list of possible assessment types
assessment_types = ['coursework', 'assessment', 'quiz', 'participation', 'exam']

# --- Read CSV Data ---
# File path for your CSV data (ensure it contains a column named 'course_code')
data_path = "/Users/callumglasgow/Downloads/year4/HACK/corseData/DATA.csv"
df = pd.read_csv(data_path)

if 'Course Code' not in df.columns:
    raise ValueError("CSV file must contain a 'course_code' column.")

unique_courses = df['Course Code'].unique()

# --- Synthetic Data Generation ---
course_rows = []       # List for course-level data rows
assessment_rows = []   # List for assessment-level data rows

# Use tqdm to display progress for processing courses
for course in tqdm(unique_courses, desc="Processing courses"):
    # Course-level properties: assign credits and semester randomly
    credits = random.choice([10, 20])
    semester = random.choice(["Semester 1", "Semester 2", "Full year"])
    
    # Base values for overall course metrics
    base_students = random.randint(50, 150)
    base_avg = random.uniform(60, 85)
    base_std = random.uniform(5, 10)
    base_median = base_avg + random.uniform(-2, 2)
    base_mode = base_avg + random.uniform(-2, 2)
    base_pass_rate = random.uniform(0.6, 1.0)
    
    # Generate course-level data for each year
    for year in years:
        students = int(base_students * random.uniform(0.95, 1.05))
        avg_mark = base_avg + random.uniform(-2, 2)
        std_val = base_std + random.uniform(-1, 1)
        median_val = base_median + random.uniform(-1, 1)
        mode_val = base_mode + random.uniform(-1, 1)
        pass_rate_val = base_pass_rate + random.uniform(-0.05, 0.05)
        pass_rate_val = max(0, min(1, pass_rate_val))  # ensure pass rate is between 0 and 1
        
        course_rows.append({
            "course_code": course,
            "credits": credits,
            "semester": semester,
            "year": year,
            "number_of_students": students,
            "average_mark": round(avg_mark, 2),
            "std_dev": round(std_val, 2),
            "median": round(median_val, 2),
            "mode": round(mode_val, 2),
            "pass_rate": round(pass_rate_val, 2)
        })
    
    # --- Assessment-Level Data Generation ---
    # Randomly select a subset of assessment types (at least 2 types)
    num_assessments = random.randint(2, len(assessment_types))
    selected_assessments = random.sample(assessment_types, num_assessments)
    
    # Generate random weights for the selected assessments that sum to 100
    raw_weights = np.random.rand(num_assessments)
    weights = (raw_weights / raw_weights.sum() * 100).round(2)
    
    # Generate synthetic metrics for each selected assessment type
    for i, assess in enumerate(selected_assessments):
        percentage_weight = weights[i]
        
        # Base values for assessment metrics
        base_avg_assess = random.uniform(60, 85)
        base_std_assess = random.uniform(5, 10)
        base_median_assess = base_avg_assess + random.uniform(-2, 2)
        base_mode_assess = base_avg_assess + random.uniform(-2, 2)
        
        for year in years:
            avg_assess = base_avg_assess + random.uniform(-2, 2)
            std_assess = base_std_assess + random.uniform(-1, 1)
            median_assess = base_median_assess + random.uniform(-1, 1)
            mode_assess = base_mode_assess + random.uniform(-1, 1)
            
            assessment_rows.append({
                "course_code": course,
                "assessment_type": assess,
                "percentage_weight": percentage_weight,
                "year": year,
                "average_mark": round(avg_assess, 2),
                "std_dev": round(std_assess, 2),
                "median": round(median_assess, 2),
                "mode": round(mode_assess, 2)
            })

# --- Create DataFrames, sort by course_code and Save to CSV ---
course_df = pd.DataFrame(course_rows)
assessment_df = pd.DataFrame(assessment_rows)

# Sort DataFrames by course_code and year so that all entries for each course are together
course_df.sort_values(by=['course_code', 'year'], inplace=True)
assessment_df.sort_values(by=['course_code', 'year'], inplace=True)

course_df.to_csv("courses_synthetic.csv", index=False)
assessment_df.to_csv("assessments_synthetic.csv", index=False)

print("Synthetic course-level data saved to 'courses_synthetic.csv'")
print("Synthetic assessment-level data saved to 'assessments_synthetic.csv'")