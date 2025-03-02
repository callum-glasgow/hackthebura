import pandas as pd

def filter_courses_by_code(course_code):
    # Read the CSV file
    df = pd.read_csv("/Users/callumglasgow/Downloads/year4/HACK/corseData/courses_synthetic.csv")
    
    # Filter rows where 'course code' equals the specified course_code and 'year' equals 2024.
    filtered_df = df[(df['Course Code'] == course_code) & (df['year'] == 2024)]
    
    return filtered_df

# Example usage:
if __name__ == "__main__":
    # Replace 'COURSE123' with the desired course code
    course_code_input = 'ACCN08010'
    result = filter_courses_by_code(course_code_input)
    
    # Display the filtered results
    print(result)