import arxiv
from urllib.request import urlretrieve
from urllib.error import URLError
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os


def fetch_and_download_papers(search_query, paper_dictionary, max_papers = 1200):

    client = arxiv.Client()

    search = arxiv.Search(
        query = search_query,
        max_results = max_papers,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)

    for r in tqdm(results, desc=f"Downloading Research Papers for the query - {search_query}"):

        file_path_pdf = f"Research_Papers/{r.get_short_id()}.pdf"

        if os.path.exists(file_path_pdf):
            tqdm.write(f"[INFO] Paper {r.get_short_id()} already exists, skipping download.")
            continue

        try:
            urlretrieve(r.pdf_url, file_path_pdf)
        except URLError as e:
            tqdm.write(f"[WARNING] {e} encountered for paper with id {r.get_short_id()}, skipping...")
            continue

        
        

        if r.get_short_id() not in paper_dictionary["ID"]:

            paper_dictionary["ID"].append(r.get_short_id())
            paper_dictionary["Title"].append(r.title)
            paper_dictionary["Categories"].append(r.categories)
            paper_dictionary["Authors"].append([author.name for author in r.authors])
            paper_dictionary["Published"].append(r.published)
            paper_dictionary["Updated"].append(r.updated)





paper_dictionary = defaultdict(list)


fetch_and_download_papers("Machine Learning", paper_dictionary)
fetch_and_download_papers("Robotics", paper_dictionary)
fetch_and_download_papers("Computer Vision", paper_dictionary)

pd.DataFrame(paper_dictionary).to_csv("Retrieval/Fetch_Papers/Research_Papers/papers_metadata.csv", index=False)