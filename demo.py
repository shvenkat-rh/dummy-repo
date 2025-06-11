# down.py

import sys
import re
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def callCodeLLama(issue_title,issue_body):
    with open('./repomix-output.md', 'r') as file:
           markdown_content = file.read()
           print(markdown_content)

def main():
    if len(sys.argv) != 3:
        print("Usage: python down.py <issue_title> <issue_body>")
        sys.exit(1)
    
    issue_title = sys.argv[1]
    issue_body = sys.argv[2]
    callCodeLLama(issue_title,issue_body)
    result = (
        "### 🔍 Issue Information Extracted by `down.py`:\n\n"
        f"**Title:** {issue_title}\n\n"
        f"**Body:** {issue_body}\n"
    )
    print(result)

if __name__ == "__main__":
    main()
