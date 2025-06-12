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
import nltk
import os

def callCodeLLama(issue_title, issue_body):
    # Read repomix output from the same directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    repomix_file = os.path.join(base_path, 'repomix-output.md')
    
    with open(repomix_file, 'r') as file:
        markdown_content = file.read()

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    ititle = issue_title.strip()
    ibody = issue_body.strip()
    full_issue_text = f"{ititle}\n{ibody}"
    issues = [full_issue_text]

    llm = ChatOllama(model="codellama:latest")

    escaped_markdown_content = markdown_content.replace("{", "{{").replace("}", "}}")

    initial_pr = (
        "You are an AI assistant who strictly answers ONLY from this document.\n\n\n"
        + escaped_markdown_content
        + "\n\n\n"
        + """
        Issue:
        {issue}

        Please provide Root cause analysis of the issue. Any Specific code fix based on the code given in the document. Also any potential side effects or considerations.

        DO NOT USE ANY EXTERNAL KNOWLEDGE OR INFORMATION. STRICTLY USE THE DOCUMENT PROVIDED ABOVE.
        """
    )

    prompt = ChatPromptTemplate.from_template(initial_pr)

    chain = (
        {"issue": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

    try:
        embedded_issues = embedding_model.embed_documents(issues)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        embedded_issues = None

    duplicates = set()
    if embedded_issues:
        duplicate_threshold = 0.85
        num_issues = len(issues)
        for i in range(num_issues):
            for j in range(i + 1, num_issues):
                try:
                    sim = cosine_similarity([embedded_issues[i]], [embedded_issues[j]])[0][0]
                    if sim > duplicate_threshold:
                        duplicates.add(j)
                except Exception as e:
                    print(f"Error calculating similarity between issues {i+1} and {j+1}: {e}")

    unique_issues = [issue for idx, issue in enumerate(issues) if idx not in duplicates]
    responses = []

    for issue_text in unique_issues:
        try:
            response = chain.invoke(input=issue_text)
            responses.append(response)
        except Exception as e:
            print(f"Failed analysis: {e}")
            continue

    return str(responses[0]) if responses else "No response"

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_path, 'issue_title.txt'), 'r') as f:
        title = f.read()

    with open(os.path.join(base_path, 'issue_body.txt'), 'r') as f:
        body = f.read()

    response = callCodeLLama(title, body)
    result = (
        "### 🔍 Issue Information Extracted by `down.py`:\n\n"
        f"**Title:** {title}\n\n"
        f"**Body:** {body}\n"
        f"**Response:** {response}\n"
    )
    print(result)

if __name__ == "__main__":
    main()
