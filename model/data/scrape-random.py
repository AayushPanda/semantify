import os
import requests
import argparse
import time
import re
from urllib.parse import urlencode

def sanitize_filename(title):
    """Convert title to safe filename"""
    filename = re.sub(r'[^\w\s-]', '', title).strip()
    filename = re.sub(r'\s+', '_', filename)
    return filename[:200] + '.txt'

def save_article(content, title, output_dir):
    """Save individual article to a file if it meets word count"""
    word_count = len(content.split())
    if word_count < 500:
        print(f"Skipped '{title}' ({word_count} words)")
        return False
    
    filename = sanitize_filename(title)
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def get_random_wikipedia_articles(num_articles, output_dir, run_to_complete):
    """Scrape random Wikipedia articles with minimum 500 words"""
    WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
    os.makedirs(output_dir, exist_ok=True)
    
    headers = {'User-Agent': 'ResearchBot/1.0 (https://example.com; contact@example.com)'}
    
    collected = 0
    batch_size = 5  # Articles per request
    attempts = 0
    max_attempts = num_articles * 3  # Prevent infinite loops
    
    while collected < num_articles and (run_to_complete or attempts < max_attempts):
        try:
            params = {
                'action': 'query',
                'generator': 'random',
                'grnnamespace': 0,
                'prop': 'extracts',
                'explaintext': 1,
                'format': 'json',
                'grnlimit': min(batch_size, num_articles - collected)
            }

            response = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            pages = data.get('query', {}).get('pages', {}).values()

            for page in pages:
                if collected >= num_articles:
                    break
                
                attempts += 1
                content = page.get('extract', '').strip()
                if not content:
                    continue
                
                title = page.get('title', f'Untitled_{attempts}')
                if save_article(content, title, output_dir):
                    collected += 1
                    print(f"Saved '{title}' ({len(content.split())} words) [{collected}/{num_articles}]")

            time.sleep(1.5)  # Respectful delay between requests

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    success_rate = (collected/num_articles)*100 if num_articles > 0 else 0
    print(f"\nSaved {collected}/{num_articles} articles ({success_rate:.1f}% success rate)")
    print(f"Total attempts made: {attempts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape Wikipedia articles with minimum 500 words')
    parser.add_argument('num_articles', type=int, help='Number of valid articles to save')
    parser.add_argument('output_dir', help='Output directory for article files')
    parser.ad_argument('run_to_complete', help='Rip and tear until it is done')
    args = parser.parse_args()

    get_random_wikipedia_articles(args.num_articles, args.output_dir, args.run_to_complete)