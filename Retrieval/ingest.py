from pypdf import PdfReader
import re


def read_pdf(pdf_file):

    reader = PdfReader(pdf_file)
    pdf_text = ""

    for page in reader.pages:

        pdf_text = pdf_text + page.extract_text() + " "

    pdf_text_clean = pdf_text.encode("utf-8", "ignore").decode("utf-8")

    pdf_text_split = re.split(r"\n?\d*\.*\s*References\n|\n?\d*\.?\s*Bibliography\n", pdf_text_clean, flags=re.IGNORECASE)

    return pdf_text_split[0]


