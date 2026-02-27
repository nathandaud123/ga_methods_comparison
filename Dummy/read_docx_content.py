import zipfile
import re
import os

docx_path = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\statistic\statistic analysis.docx"
output_path = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\docx_content.txt"

def get_docx_text(path):
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content = docx.read('word/document.xml').decode('utf-8')
            matches = re.findall(r'<w:t[^>]*>(.*?)</w:t>', xml_content)
            full_text = ' '.join(matches)
            return full_text
    except Exception as e:
        return f"Error reading docx: {str(e)}"

if __name__ == "__main__":
    if os.path.exists(docx_path):
        content = get_docx_text(docx_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Done writing to docx_content.txt")
    else:
        print(f"File not found: {docx_path}")
