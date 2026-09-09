import arxiv
from urllib.request import urlretrieve
from urllib.error import URLError
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os


def fetch_and_download_papers(search_query, paper_dictionary,metadata_path, max_papers = 1200):

    client = arxiv.Client()

    search = arxiv.Search(
        query = search_query,
        max_results = max_papers,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)
    new_paper_count = 0

    for r in tqdm(results, desc=f"Downloading Research Papers for the query - {search_query}"):

        file_path_pdf = f"Retrieval/Fetch_Papers/Research_Papers/{r.get_short_id()}.pdf"

        if not os.path.exists(file_path_pdf):
            try:
                urlretrieve(r.pdf_url, file_path_pdf)
            except URLError as e:
                tqdm.write(f"[WARNING] {e} encountered for paper with id {r.get_short_id()}, skipping...")
                continue
        else:
             tqdm.write(f"[INFO] Paper {r.get_short_id()} already exists, skipping download.")
       

        
        

        if r.get_short_id() not in paper_dictionary["ID"]:

            paper_dictionary["ID"].append(r.get_short_id())
            paper_dictionary["Title"].append(r.title)
            paper_dictionary["Categories"].append(r.categories)
            paper_dictionary["Authors"].append([author.name for author in r.authors])
            paper_dictionary["Published"].append(r.published)
            paper_dictionary["Updated"].append(r.updated)

            new_paper_count += 1

            if new_paper_count % 50 == 0:

                pd.DataFrame(paper_dictionary).to_csv(metadata_path, index=False)

    pd.DataFrame(paper_dictionary).to_csv(metadata_path, index=False)

metadata_path = "Retrieval/Fetch_Papers/Research_Papers/papers_metadata.csv"

if os.path.exists(metadata_path):
    paper_dictionary = pd.read_csv(metadata_path).to_dict(orient="list")
else:
    paper_dictionary = defaultdict(list)


fetch_and_download_papers("Machine Learning", paper_dictionary, metadata_path)
fetch_and_download_papers("Robotics", paper_dictionary,metadata_path)
fetch_and_download_papers("Computer Vision", paper_dictionary,metadata_path)
fetch_and_download_papers("Chemistry", paper_dictionary, metadata_path)
