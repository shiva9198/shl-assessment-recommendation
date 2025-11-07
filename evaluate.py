import pandas as pd
import requests
import json
from collections import defaultdict

# --- 1. CONFIGURATION ---
# PLEASE UPDATE THESE VALUES

# Path to your Excel file
DATASET_FILE_PATH = "data/gen_ai_dataset.xlsx" 

# The name of the sheet with the "answer key" (queries + ground truth URLs)
LABELED_SHEET_NAME = "Train-Set" 

# The name of the sheet with the test queries
UNLABELED_SHEET_NAME = "Test-Set" 

# The URL of your running FastAPI server
API_ENDPOINT = "http://127.0.0.1:8000/recommend"

# The name of the final CSV file for submission
SUBMISSION_FILE_NAME = "predictions.csv"

# --- 2. EVALUATION FUNCTIONS ---

def get_recommendations(query):
    """Calls your local API and returns a list of recommended URLs."""
    try:
        response = requests.post(API_ENDPOINT, json={"query": query})
        if response.status_code == 200:
            data = response.json()
            # Extract just the URLs from the response
            urls = [rec["url"] for rec in data.get("recommended_assessments", [])]
            return urls
        else:
            print(f"API Error for query '{query[:20]}...': {response.status_code}")
            return []
    except Exception as e:
        print(f"Failed to call API: {e}")
        return None

def calculate_recall_at_10(predicted_urls, ground_truth_urls):
    """
    Calculates Recall@10.
    Recall = (Number of relevant items in top 10) / (Total number of relevant items)
    
    """
    # Find the intersection of predicted and ground truth
    hits = len(set(predicted_urls) & set(ground_truth_urls))
    
    # Total number of "correct" answers for this query
    total_relevant = len(ground_truth_urls)
    
    if total_relevant == 0:
        return 1.0 if hits == 0 else 0.0 # Handle case with no ground truth
        
    return hits / total_relevant

# --- 3. MAIN EVALUATION ---

def run_evaluation():
    print("--- 🧪 Starting Evaluation Phase ---")
    
    # --- Part A: Calculate Mean Recall@10 (for your report) ---
    print(f"\n[Part A] Loading labeled data from '{LABELED_SHEET_NAME}'...")
    try:
        df_labeled = pd.read_excel(DATASET_FILE_PATH, sheet_name=LABELED_SHEET_NAME)
    except Exception as e:
        print(f"Error loading sheet: {e}. Check file path and sheet name.")
        return

    # Group the ground truth URLs by query
    ground_truth = defaultdict(list)
    for _, row in df_labeled.iterrows():
        ground_truth[row['Query']].append(row['Assessment_url'])

    unique_queries = list(ground_truth.keys())
    print(f"Found {len(unique_queries)} unique queries in the labeled set.")

    all_recalls = []
    
    for query in unique_queries:
        print(f"Testing query: '{query[:50]}...'")
        
        # 1. Get recommendations from YOUR API
        predicted_urls = get_recommendations(query)
        if predicted_urls is None:
            print("API is not running. Please start main.py in another terminal.")
            return

        # 2. Get the "correct" answers
        truth_urls = ground_truth[query]
        
        # 3. Calculate score
        recall_10 = calculate_recall_at_10(predicted_urls, truth_urls)
        all_recalls.append(recall_10)
        print(f"  -> Recall@10: {recall_10:.2f} ({len(predicted_urls)} predicted, {len(truth_urls)} ground truth)")

    # Calculate the final Mean Recall@10
    if all_recalls:
        mean_recall = sum(all_recalls) / len(all_recalls)
        print("\n-------------------------------------------------")
        print(f"✅ FINAL SCORE: Mean Recall@10 = {mean_recall:.4f}")
        print("-------------------------------------------------")
        print("Use this score in your 2-page report.")
    else:
        print("No queries were processed. Cannot calculate Mean Recall.")

    # --- Part B: Generate Submission CSV (for the unlabeled set) ---
    print(f"\n[Part B] Loading unlabeled data from '{UNLABELED_SHEET_NAME}'...")
    try:
        df_unlabeled = pd.read_excel(DATASET_FILE_PATH, sheet_name=UNLABELED_SHEET_NAME)
        unlabeled_queries = df_unlabeled['Query'].unique()
    except Exception as e:
        print(f"Error loading sheet: {e}. Check file path and sheet name.")
        return
        
    print(f"Found {len(unlabeled_queries)} queries for submission.")
    
    submission_data = [] # List of [query, url] pairs
    
    for query in unlabeled_queries:
        print(f"Generating predictions for: '{query[:50]}...'")
        
        # Get recommendations from YOUR API
        predicted_urls = get_recommendations(query)
        
        if not predicted_urls:
            print("  -> Warning: API returned no recommendations for this query.")
            
        # Add each [query, recommendation_url] pair, as required by the PDF
        for url in predicted_urls:
            submission_data.append([query, url])

    # Create the submission DataFrame and save to CSV
    df_submission = pd.DataFrame(submission_data, columns=['Query', 'Assessment_url'])
    df_submission.to_csv(SUBMISSION_FILE_NAME, index=False)
    
    print("\n-------------------------------------------------")
    print(f"✅ SUBMISSION FILE CREATED: {SUBMISSION_FILE_NAME}")
    print("-------------------------------------------------")
    print("This file is ready to be submitted.")
    print("\n--- 🏁 Evaluation Complete ---")

# --- 4. RUN THE SCRIPT ---
if __name__ == "__main__":
    print("Make sure your API server (main.py) is running in another terminal.")
    print("Press Enter to continue...")
    input()
    run_evaluation()