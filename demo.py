import sys
import os
import re
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

def callCodeLLama(issue_title, issue_body):
    with open('./repomix-output.md', 'r') as file:
        markdown_content = file.read()

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    ititle = issue_title.strip()
    ibody = issue_body.strip()
    full_issue_text = f"{ititle}\n{ibody}"
    issues = [full_issue_text]

    llm = ChatOllama(model="codellama:latest")

    escaped_markdown_content = markdown_content.replace("{", "{{").replace("}", "}}")

    '''initial_pr = (
        "You are an AI assistant who strictly answers ONLY from this document.\n\n\n"
        + escaped_markdown_content
        + "\n\n\n"
        + """
        Issue:
        {issue}
        
        Please provide Root cause analysis of the issue. Any Specific code fix based on the code given in the document.
        Also any potential side effects or considerations.
        
        DO NOT USE ANY EXTERNAL KNOWLEDGE OR INFORMATION. STRICTLY USE THE DOCUMENT PROVIDED ABOVE.
        """
    )'''

    initial_pr = """You are an AI assistant who strictly answers ONLY from this document.

    CRITICAL INSTRUCTION: You MUST categorize every response as exactly one of: Bug, Enhancement, Documentation, Question.
    
    """ + "\n\n\n"+escaped_markdown_content + """\n\n\n""" +"""
    
    Issue:
    {issue}
    
    Provide your response in this exact format:
    
    1. **Root Cause Analysis:** [Your analysis]
    2. **Code Fix:** [Your fix based on the document]
    3. **Side Effects/Considerations:** [Your considerations if any]
    4. **Issue Category:** [REQUIRED - Must be exactly one of: Bug, Enhancement, Documentation, Question]
    
    FINAL REMINDER: Do not forget to specify the issue category. This is mandatory.
    
    DO NOT USE ANY EXTERNAL KNOWLEDGE OR INFORMATION. STRICTLY USE THE DOCUMENT PROVIDED ABOVE."""

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
        threshold = 0.85
        for i in range(len(issues)):
            for j in range(i + 1, len(issues)):
                try:
                    sim = cosine_similarity([embedded_issues[i]], [embedded_issues[j]])[0][0]
                    if sim > threshold:
                        duplicates.add(j)
                except Exception as e:
                    print(f"Error calculating similarity: {e}")

    unique_issues = [issue for idx, issue in enumerate(issues) if idx not in duplicates]
    responses = []

    for issue_text in unique_issues:
        try:
            response = chain.invoke(input=issue_text)
            responses.append(response)
        except Exception as e:
            responses.append(f"Error analyzing issue: {e}")

    return responses[0] if responses else "No response"


def main():
    if len(sys.argv) != 3:
        print("Usage: python down.py <issue_title_file> <issue_body_file>")
        sys.exit(1)

    issue_title = sys.argv[1]
    issue_body = sys.argv[2]
    
    response = callCodeLLama(issue_title, issue_body)
    result = (
        "### Root Cause Analysis and Triage by AnsibleLLM:\n\n"
        f"**Response:** {response}\n"
    )
    print(result)


if __name__ == "__main__":
    main()
