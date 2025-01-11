import json
import os
import requests
from time import sleep
import xml.etree.ElementTree as ET

from tqdm import tqdm
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


def parse_results(feeds):
    data = []
    for feed in tqdm(feeds):
        try:
            root = ET.fromstring(feed)
            titles = root.findall(".//hit/info/title")
            years = root.findall(".//hit/info/year")
            authors = root.findall(".//hit/info/authors")
            entries = zip(titles, years, authors)
            for entry in entries:
                title = entry[0].text.lower()
                if any([kw in title for kw in [" ai", "ai ", "intelligen", 'machine learning']]):
                    data.append({
                        'title': entry[0].text.replace(';', '$$$$$$$$'),
                        'year': int(entry[1].text) or -1,
                        'author': ' and ' .join([e.text for e in entry[2].findall("author")]).replace(';', '$$$$$$$$')
                    })
        except ET.ParseError:
            print("Error parsing XML feed.")
    return pd.DataFrame(data).sort_values(by='year')
                


                # added = {}
                # if not (" ai " in title or "intelligen" in title or 'machine learning' in title):
                #     removed_titles.append(title)
                #     continue
                # for keyword, family in keywords.items():
                #     if family not in added:
                #         if title.find(keyword) >= 0:
                #             title = title.replace(keyword, keyword.upper())
                #             added[family] = True
                #             if year not in keyword_counts[family]:
                #                 keyword_counts[family][year] = 0
                #             keyword_counts[family][year] += 1
                # if len(added) == 0:
                #     removed_titles.append(title)
                # else:
                #     if year not in all_titles:
                #         all_titles[year] = []
                #     all_titles[year].append(title)

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

    journals = query_papers(keywords, type='Journal_Articles')
    conferences = query_papers(keywords, type='Conference_and_Workshop_Papers')
    books = query_papers(keywords, type='Parts_in_Books_or_Collections')
    all_results = journals + conferences + books
    data = parse_results(all_results)
    data.to_csv(os.path.join(os.path.dirname(__file__), "parse_dblp_data.csv"), index=False, sep=';')
    assert ';' not in data.to_string()

    # dblp_counts, dblp_titles, removed_titles = parse_results(all_results, keywords, logic='dblp')

    # file with counts / summary
    # dblp_df = pd.DataFrame(dblp_counts).sort_index()
    
    # 
    # with open(os.path.join(os.path.dirname(__file__), "dblp_titles.csv"), 'w') as jf:
    #     json.dump(dblp_titles, jf, indent=4)
