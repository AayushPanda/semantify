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

def get_wiki_from_titles(titleList, output_dir):
    """Scrape article content from urls in a file"""
    WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
    os.makedirs(output_dir, exist_ok=True)
    
    headers = {'User-Agent': 'ResearchBot/1.0 (https://example.com; contact@example.com)'}
    
    collected = 0
    attempts = 0
    titles = open(titleList, 'r').readlines()
    num_articles = len(titles)
    for title in titles:
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "titles": title.strip(),
                "formatversion": "2",
                "exlimit": "max",
                "explaintext": 1
            }

            response = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            pages = data.get('query', {}).get('pages', {})

            for page in pages:
                
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
    parser = argparse.ArgumentParser(description='Scrape Wikipedia articles from urls in a file')
    parser.add_argument('urllist', help='Input file with urls')
    parser.add_argument('output_dir', help='Output directory for article files')
    args = parser.parse_args()

    get_wiki_from_titles(args.urllist, args.output_dir)