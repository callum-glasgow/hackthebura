import requests

BASE_URL = "http://127.0.0.1:8000/search"

def test_options():
    headers = {
        "Origin": "http://127.0.0.1",  # Match the URL host exactly
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"
    }
    response = requests.options(BASE_URL, headers=headers)
    print("OPTIONS Request:")
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)

def test_full_index_search():
    payload = {
        "query": "machine learning course",
        "year": ""  # Use full index if empty or 'all'
    }
    headers = {
        "Content-Type": "application/json",
        "Origin": "http://127.0.0.1"  # Use exact match
    }
    response = requests.post(BASE_URL, json=payload, headers=headers)
    print("\nFull Index Search:")
    print("Status Code:", response.status_code)
    try:
        data = response.json()
        print("Response JSON:", data)
    except Exception as e:
        print("Failed to decode JSON. Response text:")
        print(response.text)
        print("Error:", e)

def test_year_specific_search():
    payload = {
        "query": "data science",
        "year": "3"  # Specific Programme Year
    }
    headers = {
        "Content-Type": "application/json",
        "Origin": "http://127.0.0.1"
    }
    response = requests.post(BASE_URL, json=payload, headers=headers)
    print("\nYear-Specific Search:")
    print("Status Code:", response.status_code)
    try:
        data = response.json()
        print("Response JSON:", data)
    except Exception as e:
        print("Failed to decode JSON. Response text:")
        print(response.text)
        print("Error:", e)

if __name__ == "__main__":
    test_options()
    test_full_index_search()
    test_year_specific_search()