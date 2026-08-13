from pypdf import PdfReader
import re


def read_pdf(pdf_file):

    reader = PdfReader(pdf_file)
    pdf_text = ""

    for page in reader.pages:

        pdf_text = pdf_text + page.extract_text() + " "

    pdf_text_split = re.split(r"\n?\d*\.*\s*References\n|\n\d*\.?\s*Bibliography\n", pdf_text, flags=re.IGNORECASE)

    return pdf_text_split[0]

# print(read_pdf("Fetch_Papers/Research_Papers/2608.06332v1.pdf"))
