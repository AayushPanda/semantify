import pymupdf  # PyMuPDF
import docx

def pdf_to_txt(pdf_path, txt_path):
    # Open the PDF file
    doc = pymupdf.open(pdf_path)
    
    # Open the output text file
    with open(txt_path, "w", encoding="utf-8") as output_file:
        # Iterate over each page
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)  # Load page
            text = page.get_text()  # Extract text from page
            
            # Write extracted text to the text file
            output_file.write(text)
            output_file.write("\n\n")  # Add space between pages

    print(f"PDF has been converted to text and saved at: {txt_path}")

def docx_to_txt(docx_path, txt_path):
    # Open the DOCX file
    doc = docx.Document(docx_path)
    
    # Open the output text file
    with open(txt_path, "w", encoding="utf-8") as output_file:
        # Iterate over each paragraph
        for para in doc.paragraphs:
            text = para.text  # Extract text from paragraph
            
            # Write extracted text to the text file
            output_file.write(text)
            output_file.write("\n")  # Add space between paragraphs

    print(f"DOCX has been converted to text and saved at: {txt_path}")

if __name__ == "__main__":
    # Test
    # Convert PDF to text
    pdf_path = "sample.pdf"
    txt_path = "sample.txt"
    pdf_to_txt(pdf_path, txt_path)
    
    # Convert DOCX to text
    docx_path = "sample.docx"
    txt_path = "sample_docx.txt"
    docx_to_txt(docx_path, txt_path)


