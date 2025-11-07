import csv
import pandas as pd
import os
import requests
import json
from typing import List

# --- Configuration ---

# 1. Point to the test dataset
UNLABELED_TEST_SET_FILE = 'data/gen_ai_dataset.xlsx'

# 2. Final submission file
SUBMISSION_FILE_OUTPUT = 'final_submission.csv'

# 3. API endpoint for the 0.5133 baseline system
API_ENDPOINT = 'http://127.0.0.1:8000/recommend'

# --- Your Baseline Model Integration ---

def get_recommendations_from_baseline(query: str) -> List[str]:
    """
    This function integrates with the 0.5133 baseline API to get recommendations.
    
    Returns a list of recommended assessment URLs (as strings).
    """
    try:
        print(f"Processing query: {query[:50]}...")
        
        # Call the baseline API
        response = requests.post(
            API_ENDPOINT,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            recommended_urls = []
            
            # Extract URLs from the API response
            for assessment in data.get('recommended_assessments', []):
                url = assessment.get('url', '')
                if url:
                    recommended_urls.append(url)
            
            print(f"  -> Got {len(recommended_urls)} recommendations")
            return recommended_urls
        else:
            print(f"  -> API Error: {response.status_code}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"  -> Request Error: {e}")
        return []
    except Exception as e:
        print(f"  -> Unexpected Error: {e}")
        return []

# --- Main CSV Generation Logic ---

def load_test_queries(filename: str) -> List[str]:
    """
    Loads queries from the unlabeled test set Excel file.
    """
    queries = []
    try:
        # Read the Test-Set sheet from Excel
        df = pd.read_excel(filename, sheet_name='Test-Set')
        queries = df['Query'].tolist()
        print(f"Loaded {len(queries)} queries from {filename}")
        return queries
        
    except FileNotFoundError:
        print(f"Error: Test set file not found at {filename}")
        return []
    except Exception as e:
        print(f"Error loading test queries: {e}")
        return []

def create_submission_csv():
    """
    Runs the baseline model on all test queries and writes
    the results to the submission CSV.
    """
    print("Starting CSV generation for final submission...")
    print("=" * 60)
    
    # Load test queries
    test_queries = load_test_queries(UNLABELED_TEST_SET_FILE)
    
    if not test_queries:
        print("No queries loaded. Exiting.")
        return

    print(f"Processing {len(test_queries)} test queries...")
    print("=" * 60)

    # This list will hold all the rows for the CSV
    csv_rows = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Processing query:")
        print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        # Get recommendations from the baseline model
        try:
            recommended_urls = get_recommendations_from_baseline(query)
            
            if not recommended_urls:
                print(f"Warning: No recommendations found for query {i}")
                continue
                
            # Add one row for each recommended URL
            for url in recommended_urls:
                csv_rows.append({'Query': query, 'Assessment_url': url})
                
            print(f"Added {len(recommended_urls)} recommendations to CSV")
                
        except Exception as e:
            print(f"Error processing query {i}: {e}")

    print("\n" + "=" * 60)
    print(f"Processing complete. Total recommendations: {len(csv_rows)}")

    # Write all results to the CSV file
    try:
        with open(SUBMISSION_FILE_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            # Define the exact headers required
            fieldnames = ['Query', 'Assessment_url']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write the header row
            writer.writeheader()
            
            # Write all the data rows
            writer.writerows(csv_rows)
            
        print(f"\n✅ Successfully created submission file: {SUBMISSION_FILE_OUTPUT}")
        print(f"📊 Total rows written: {len(csv_rows)}")
        
        # Show summary statistics
        unique_queries = len(set(row['Query'] for row in csv_rows))
        print(f"📝 Unique queries processed: {unique_queries}")
        print(f"🔗 Average recommendations per query: {len(csv_rows)/unique_queries:.1f}")

    except IOError as e:
        print(f"Error writing to CSV file {SUBMISSION_FILE_OUTPUT}: {e}")

def check_api_status():
    """
    Check if the baseline API is running and accessible.
    """
    try:
        response = requests.get('http://127.0.0.1:8000/health', timeout=5)
        if response.status_code == 200:
            print("✅ Baseline API is running and accessible")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ Baseline API is not running. Please start main.py first.")
        print("   Command: python main.py")
        return False

# --- Run the script ---
if __name__ == "__main__":
    print("SHL Assessment Recommendation - Final Submission Generator")
    print("=" * 60)
    
    # Check if API is running
    if not check_api_status():
        print("\nPlease start the baseline API server first:")
        print("1. Open a new terminal")
        print("2. Run: cd /Users/shiva/Documents/SHL")
        print("3. Run: source .venv/bin/activate")
        print("4. Run: export OPENROUTER_API_KEY='your-api-key'")
        print("5. Run: python main.py")
        print("6. Then run this script again")
        exit(1)
    
    # Generate the submission CSV
    create_submission_csv()
    
    print("\n" + "=" * 60)
    print("🎯 Submission generation complete!")
    print(f"📁 File ready for submission: {SUBMISSION_FILE_OUTPUT}")
    print("=" * 60)