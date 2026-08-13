from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pdf_text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap= 200)

    chunks = text_splitter.split_text(pdf_text)

    return chunks