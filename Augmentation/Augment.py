

def augment_query_results(query_results):

    context = ""
    for i, (document, metadata) in enumerate(query_results.values()):
        single_line_document = document.replace("\n", " ")
        context += f"TITLE {i+1} : {metadata['Title']}\n\nCONTEXT {i+1} : {single_line_document}\n\n\n"

    return context