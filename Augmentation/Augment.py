

def augment_query_results(query_results):

    context = ""
    for document, metadata in query_results:
        single_line_document = document.replace("\n", " ")
        context += f"{metadata['Title']}\n{single_line_document}\n"

    return context