import arxiv
from urllib.request import urlretrieve
import pandas as pd
from collections import defaultdict

def fetch_and_download_papers(search_query, paper_dictionary, max_papers = 5):

    client = arxiv.Client()

    search = arxiv.Search(
        query = search_query,
        max_results = max_papers,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)

    for r in results:

        file_path_pdf = f"Research_Papers/{r.get_short_id()}.pdf"
    
        urlretrieve(r.pdf_url, file_path_pdf)

        if r.get_short_id() not in paper_dictionary["ID"]:

            paper_dictionary["ID"].append(r.get_short_id())
            paper_dictionary["Title"].append(r.title)
            paper_dictionary["Categories"].append(r.categories)
            paper_dictionary["Authors"].append([author.name for author in r.authors])
            paper_dictionary["Published"].append(r.published)
            paper_dictionary["Updated"].append(r.updated)





paper_dictionary = defaultdict(list)


fetch_and_download_papers("machine learning", paper_dictionary)
fetch_and_download_papers("Robotics", paper_dictionary)
fetch_and_download_papers("Neural ODE", paper_dictionary)

pd.DataFrame(paper_dictionary).to_csv("Research_Papers/papers_metadata.csv", index=False)