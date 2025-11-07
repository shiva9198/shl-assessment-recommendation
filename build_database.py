import os
import shutil
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

# --- 1. CONFIGURATION ---
DATASET_FILE_PATH = "data/gen_ai_dataset.xlsx" # The "answer key" Excel file
LABELED_SHEET_NAME = "Train-Set"                # The sheet with the answers
DB_PERSIST_DIRECTORY = "chroma_db_shl"          # Where the new DB will be saved

# --- 2. SCRAPING FUNCTION ---
def get_session():
    """Create a session with enhanced headers to bypass browser detection"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    })
    return session

def scrape_product_page(url, session):
    """
    Scrapes a single SHL product page for its name, description, and test type.
    If scraping fails, creates a document from URL metadata.
    """
    try:
        print(f"    Fetching: {url}")
        response = session.get(url, timeout=15)
        if not response.ok:
            print(f"  -> HTTP Error {response.status_code} for {url}")
            return create_fallback_document(url)
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check for browser warning (our previous issue)
        if "We recommend upgrading to a modern browser" in response.text:
            print(f"  -> Browser detection warning, using fallback for {url}")
            return create_fallback_document(url)
        
        # Extract product information with multiple fallback strategies
        name = extract_product_name(soup, url)
        description = extract_product_description(soup)
        test_types = extract_test_types(soup, description)
        
        # Store all data in metadata, as required by the API
        metadata = {
            "source": url,
            "url": url,  # Important: This must match the ground truth URLs
            "name": name,
            "description": description,
            "test_type": ", ".join(test_types), # Convert list to string for Chroma
            "adaptive_support": "No",  # Placeholder
            "duration": 0,           # Placeholder  
            "remote_support": "Yes"  # Placeholder
        }
        
        # Create the searchable content
        page_content = f"Product Name: {name}\n" \
                       f"Test Type: {metadata['test_type']}\n" \
                       f"Description: {description}"
        
        print(f"  -> Success: {name}")
        return Document(page_content=page_content, metadata=metadata)

    except Exception as e:
        print(f"  -> EXCEPTION scraping {url}, using fallback: {e}")
        return create_fallback_document(url)

def create_fallback_document(url):
    """
    Creates a document from URL metadata when scraping fails
    """
    # Extract product info from URL structure
    url_parts = url.split('/')
    if 'view' in url_parts:
        view_index = url_parts.index('view')
        if view_index + 1 < len(url_parts):
            product_slug = url_parts[view_index + 1]
            # Clean up the slug to create a readable name
            name = product_slug.replace('-', ' ').replace('%28', '(').replace('%29', ')').title()
            name = name.replace('New', '').strip()
            if not name:
                name = "SHL Assessment"
        else:
            name = "SHL Assessment"
    else:
        name = "SHL Assessment"
    
    # Infer test type from URL and name
    url_lower = url.lower()
    name_lower = name.lower()
    test_types = []
    
    # Technical skills
    if any(keyword in url_lower or keyword in name_lower for keyword in 
           ['java', 'python', 'javascript', 'css', 'html', 'sql', 'selenium', 'drupal', 'tableau', 'excel']):
        test_types.append("Knowledge & Skills")
    
    # Personality/behavior
    if any(keyword in url_lower or keyword in name_lower for keyword in 
           ['personality', 'opq', 'leadership', 'communication', 'interpersonal', 'sales']):
        test_types.append("Personality & Behaviour")
    
    # Cognitive
    if any(keyword in url_lower or keyword in name_lower for keyword in 
           ['verify', 'numerical', 'verbal', 'reasoning', 'inductive']):
        test_types.append("Cognitive")
        
    if not test_types:
        test_types = ["Assessment"]
    
    # Create description from inferred information
    description = f"SHL {name} assessment. "
    if "Knowledge & Skills" in test_types:
        description += "Tests technical knowledge and practical skills. "
    if "Personality & Behaviour" in test_types:
        description += "Evaluates personality traits and behavioral competencies. "
    if "Cognitive" in test_types:
        description += "Measures cognitive abilities and reasoning skills. "
    
    metadata = {
        "source": url,
        "url": url,
        "name": name,
        "description": description,
        "test_type": ", ".join(test_types),
        "adaptive_support": "No",
        "duration": 0,
        "remote_support": "Yes"
    }
    
    page_content = f"Product Name: {name}\n" \
                   f"Test Type: {metadata['test_type']}\n" \
                   f"Description: {description}"
    
    print(f"  -> Fallback created: {name}")
    return Document(page_content=page_content, metadata=metadata)

def extract_product_name(soup, url):
    """Extract product name with multiple fallback strategies"""
    # Strategy 1: Standard product title
    name_selectors = [
        'h1.product-title__heading',
        'h1.product-header__title', 
        'h1[data-testid="product-title"]',
        '.product-title h1',
        '.page-title h1',
        'h1'
    ]
    
    for selector in name_selectors:
        name_tag = soup.select_one(selector)
        if name_tag:
            name = name_tag.get_text(strip=True)
            if name and len(name) > 3:  # Valid name
                return name
    
    # Fallback: Extract from URL
    url_parts = url.split('/')
    if url_parts:
        return url_parts[-1].replace('-', ' ').title()
    
    return "Unknown Product"

def extract_product_description(soup):
    """Extract product description with multiple fallback strategies"""
    # Strategy 1: Product description sections
    desc_selectors = [
        '.product-description__text',
        '.product-overview',
        '.product-details',
        '.description',
        '.overview',
        'meta[name="description"]'
    ]
    
    for selector in desc_selectors:
        if selector.startswith('meta'):
            desc_tag = soup.select_one(selector)
            if desc_tag and desc_tag.get('content'):
                return desc_tag['content'].strip()
        else:
            desc_tag = soup.select_one(selector)
            if desc_tag:
                description = desc_tag.get_text(strip=True)
                if description and len(description) > 10:
                    return description
    
    # Fallback: Get text from main content areas
    content_selectors = ['.main-content', '.content', 'main', 'article']
    for selector in content_selectors:
        content_tag = soup.select_one(selector)
        if content_tag:
            paragraphs = content_tag.find_all('p')
            if paragraphs:
                description = ' '.join([p.get_text(strip=True) for p in paragraphs[:3]])
                if len(description) > 20:
                    return description
    
    return "No description available."

def extract_test_types(soup, description):
    """Extract test types with multiple strategies"""
    test_types = []
    
    # Strategy 1: Look for specific test type indicators
    text_content = (description + " " + soup.get_text()).lower()
    
    # Knowledge & Skills indicators
    if any(keyword in text_content for keyword in ['java', 'python', 'programming', 'coding', 'technical', 'skill', 'knowledge', 'aptitude', 'ability']):
        test_types.append("Knowledge & Skills")
    
    # Personality & Behaviour indicators  
    if any(keyword in text_content for keyword in ['personality', 'behaviour', 'behavior', 'leadership', 'communication', 'teamwork', 'motivation', 'trait']):
        test_types.append("Personality & Behaviour")
    
    # Cognitive indicators
    if any(keyword in text_content for keyword in ['cognitive', 'reasoning', 'logical', 'numerical', 'verbal', 'intelligence']):
        test_types.append("Cognitive")
    
    # Default fallback
    if not test_types:
        test_types = ["Assessment"]
    
    return test_types

# --- 3. MAIN SCRIPT ---
def main():
    print("--- 🚀 Starting Database Build ---")
    
    # Check for API key
    if "OPENROUTER_API_KEY" not in os.environ:
        print("❌ Error: OPENROUTER_API_KEY not found. Please set it in your environment.")
        return
    
    # --- Step A: Load the "Answer Key" ---
    try:
        df = pd.read_excel(DATASET_FILE_PATH, sheet_name=LABELED_SHEET_NAME)
        unique_urls = df['Assessment_url'].unique()
        print(f"✅ Found {len(unique_urls)} unique product URLs in the answer key.")
    except Exception as e:
        print(f"❌ Error: Could not read {DATASET_FILE_PATH}. Make sure the path and sheet name are correct.")
        print(e)
        return

    # --- Step B: Scrape All Products ---
    session = get_session()
    documents = []
    failed_urls = []
    
    print(f"🔍 Scraping {len(unique_urls)} product pages...")
    for i, url in enumerate(unique_urls):
        print(f"Progress: {i+1}/{len(unique_urls)}")
        doc = scrape_product_page(url, session)
        if doc:
            documents.append(doc)
        else:
            failed_urls.append(url)
        time.sleep(1)  # Be polite to the server

    print(f"\n✅ Successfully scraped {len(documents)} products.")
    if failed_urls:
        print(f"⚠️  Failed to scrape {len(failed_urls)} URLs:")
        for url in failed_urls[:5]:  # Show first 5 failures
            print(f"    {url}")
    
    if not documents:
        print("❌ Error: No documents were scraped. Check selectors and network.")
        return

    # --- Step C: Build the Vector Database ---
    print("🗑️ Deleting old database (if it exists)...")
    if os.path.exists(DB_PERSIST_DIRECTORY):
        shutil.rmtree(DB_PERSIST_DIRECTORY)

    print("📊 Creating new vector database... (this may take a moment)")
    try:
        embedding = OpenAIEmbeddings(
            model="openai/text-embedding-ada-002",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.environ["OPENROUTER_API_KEY"]
        )
        
        vectordb = Chroma.from_documents(
            documents=documents, 
            embedding=embedding, 
            persist_directory=DB_PERSIST_DIRECTORY
        )
        
        print("\n--- ✅ Database Build Complete! ---")
        print(f"Successfully saved {len(documents)} products to '{DB_PERSIST_DIRECTORY}'.")
        print("You are now ready to run 'main.py' and 'evaluate.py'.")
        
        # Quick test to verify the database
        print("\n🧪 Quick database test:")
        test_results = vectordb.similarity_search("Java programming", k=3)
        print(f"Found {len(test_results)} results for 'Java programming' query")
        
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        return

if __name__ == "__main__":
    main()