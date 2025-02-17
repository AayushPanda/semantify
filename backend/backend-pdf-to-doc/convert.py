import fitz  # PyMuPDF

def pdf_to_txt(pdf_path, txt_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
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

# Specify paths
pdf_file = 'C:/Users/ricke/Documents/GitHub/semantify/backend-pdf-to-doc/sample.pdf'
txt_file = 'C:/Users/ricke/Documents/GitHub/semantify/backend-pdf-to-doc/output.txt'

# Call the conversion function
pdf_to_txt(pdf_file, txt_file)
