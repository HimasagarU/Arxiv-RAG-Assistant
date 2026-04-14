import sqlite3
import os
import requests
import feedparser
import time

DB_PATH = "data/arxiv_papers.db"

def clean_text(text):
    return " ".join(text.split())

def extract_id(entry_id):
    return entry_id.split("/abs/")[-1].split("v")[0]

def main():
    if not os.path.exists(DB_PATH):
        return

    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    
    needed = 20000 - count
    if needed <= 0:
        if needed < 0:
            excess = abs(needed)
            print(f"Trimming {excess} papers to hit exactly 20000...")
            conn.execute(f"DELETE FROM papers WHERE paper_id IN (SELECT paper_id FROM papers ORDER BY published ASC LIMIT {excess})")
            conn.commit()
            print("Successfully trimmed to 20000.")
        else:
            print("Already at exactly 20000 papers!")
        conn.close()
        return

    print(f"Database has {count} papers. Fetching EXACTLY {needed} new papers by bypassing recent overlaps...")
    
    # ArXiv API settings
    query = "cat:cs.RO"  # Robotics has very little overlap with AI/LG
    start_offset = 0  # Start from the newest, but since it's Robotics, they will be distinct
    
    papers_added = 0
    while papers_added < needed:
        fetch_count = min(100, needed - papers_added)
        url = f"http://export.arxiv.org/api/query?search_query={query}&start={start_offset}&max_results={fetch_count}&sortBy=submittedDate&sortOrder=descending"
        
        try:
            resp = requests.get(url, timeout=10)
            feed = feedparser.parse(resp.text)
        except Exception as e:
            print(f"API Error: {e}, retrying...")
            time.sleep(3)
            continue
            
        if not feed.entries:
            start_offset += 100
            continue

        for entry in feed.entries:
            if papers_added >= needed:
                break
                
            paper_id = extract_id(entry.id)
            title = clean_text(entry.get("title", ""))
            abstract = clean_text(entry.get("summary", ""))
            authors = ", ".join(a.get("name", "") for a in entry.get("authors", []))
            categories = ", ".join(t.get("term", "") for t in entry.get("tags", []))
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            pub = entry.get("published", "")
            
            # Insert and ignore duplicates
            try:
                conn.execute(
                    "INSERT INTO papers (paper_id, title, abstract, authors, categories, pdf_url, published, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (paper_id, title, abstract, authors, categories, pdf_url, pub, pub)
                )
                conn.commit()
                papers_added += 1
                if papers_added % 10 == 0:
                    print(f"Added {papers_added}/{needed} distinct papers...")
            except sqlite3.IntegrityError:
                pass  # Ignore if it still overlaps
        
        start_offset += len(feed.entries)
        time.sleep(3)

    new_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()
    print("=" * 40)
    print(f"SUCCESS: Database now contains exactly {new_count} papers.")
    print("=" * 40)
    print("Important: Now you must run 'scripts\\rebuild_indexes.bat' so your vector/Chroma indexes match the new 20,000 papers!")

if __name__ == "__main__":
    main()
