

def augment_query_results(query_results):

    context = ""
    for document, metadata, distance in query_results.values():
        single_line_document = document.replace("\n", " ")
        context += f"{metadata['Title']}\n{single_line_document}\n\n"

    return context