import json
import os
import requests
from time import sleep
import xml.etree.ElementTree as ET

import pandas as pd


def generate_query(keywords, batch, batch_size, type):
    base_url = "https://dblp.org/search/publ/api"
    params = {
        "q": 'ai|intellig|learning ' + '|'.join(kw for kw in keywords.keys()) + f' type:{type}:',
        "h": batch_size,
        "f": batch * batch_size,  # Max results per query
        "format": "xml"
    }
    return base_url, params
    

def query_papers(keywords, type, max_batches=100, batch_size=1000):
    print(f"Querying DBLP {type} for keywords...")
    results = []
    for batch in range(max_batches):
        base_url, params = generate_query(keywords, batch, batch_size, type)
        print(f"Fetching batch {batch + 1}...")  # Debug
        sleep(8)
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Failed to query API. Status code: {response.status_code}")
            break

        results.append(response.text)
        # Stop if no more results are returned
        if len(response.text) < 3000: # Assuming small feed size indicates no results
            print("No more results to fetch.")
            break

    return results


def parse_results(feeds, keywords, logic='arxiv'):
    print(f"Parsing {logic} results...")
    keyword_counts = {kw: {} for kw in keywords.values()}
    all_titles, removed_titles = {}, []

    try:
        for feed in feeds:
            root = ET.fromstring(feed)
            if logic == 'arxiv':
                entries = root.findall("atom:entry", {"atom": "http://www.w3.org/2005/Atom"})
            else: # dblp:
                titles = root.findall(".//hit/info/title")
                years = root.findall(".//hit/info/year")
                entries = zip(titles, years)
            for entry in entries:
                # extract title
                if logic == 'arxiv':
                    title = entry.find("atom:title", {"atom": "http://www.w3.org/2005/Atom"}).text.lower() or ""
                    year = 'n.a.'
                else: # dblp:
                    title = entry[0].text.lower() or ""
                    year = int(entry[1].text.lower()) or ""
                added = {}
                if not (" ai " in title or "intelligen" in title or 'machine learning' in title):
                    removed_titles.append(title)
                    continue
                for keyword, family in keywords.items():
                    if family not in added:
                        if title.find(keyword) >= 0:
                            title = title.replace(keyword, keyword.upper())
                            added[family] = True
                            if year not in keyword_counts[family]:
                                keyword_counts[family][year] = 0
                            keyword_counts[family][year] += 1
                if len(added) == 0:
                    removed_titles.append(title)
                else:
                    if year not in all_titles:
                        all_titles[year] = []
                    all_titles[year].append(title)
    except ET.ParseError:
        print("Error parsing XML feed.")

    return keyword_counts, all_titles, removed_titles

if __name__ == "__main__":

    keywords = {
        "explain": "Explainability",
        "xai": "Explainability",
        "ethic": "Ethics",
        "trustworth" : "Trustworthiness",
        "responsib": "Responsibility",
        "account": "Accountability",
        "sustain": "Sustainability",
    }

    journals = query_papers(keywords, type='Journal_Articles', max_batches=2)
    conferences = query_papers(keywords, type='Conference_and_Workshop_Papers', max_batches=2)
    books = query_papers(keywords, type='Parts_in_Books_or_Collections', max_batches=2)
    all_results = journals + conferences + books
    dblp_counts, dblp_titles, removed_titles = parse_results(all_results, keywords, logic='dblp')

    # print("Keyword occurrences in ArXiv publications:")
    # for keyword, count in arxiv_counts.items():
    #     print(f"{keyword}: {count}")

    # file with counts / summary
    dblp_df = pd.DataFrame(dblp_counts).sort_index()
    dblp_df.to_csv(os.path.join(os.path.dirname(__file__), "dblp_counts.csv"))

    # 
    with open(os.path.join(os.path.dirname(__file__), "dblp_titles.csv"), 'w') as jf:
        json.dump(dblp_titles, jf, indent=4)
