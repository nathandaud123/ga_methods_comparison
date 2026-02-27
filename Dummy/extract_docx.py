import zipfile
import xml.etree.ElementTree as ET
import os

docx_path = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\Manuscript.docx"
output_txt = "manuscript_content.txt"

def extract_text_from_docx(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            
            # XML namespace for Word
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text_parts = []
            for p in tree.findall('.//w:p', namespaces):
                paragraph_text = []
                for t in p.findall('.//w:t', namespaces):
                    if t.text:
                        paragraph_text.append(t.text)
                if paragraph_text:
                    text_parts.append("".join(paragraph_text))
            
            return "\n".join(text_parts)
            
    except Exception as e:
        return f"Error reading docx: {e}"

text = extract_text_from_docx(docx_path)
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Extracted {len(text)} characters to {output_txt}")
