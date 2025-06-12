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

def callCodeLLama(issue_title,issue_body):
    with open('./repomix-output.md', 'r') as file:
           markdown_content = file.read()

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    ititle=issue_title.strip()
    ibody=issue_body.strip()
    full_issue_text = f"{ititle}\n{ibody}"
    issues=[]
    issues.append(full_issue_text)
    llm = ChatOllama(model="codellama:latest")
    # Assuming markdown_content is already loaded (from your previous steps)
    # Escape curly braces in markdown_content
    escaped_markdown_content = markdown_content.replace("{", "{{").replace("}", "}}")
    
    # Your initial prompt
    initial_pr = """You are an AI assistant who strictly answers ONLY from this document.""" + "\n\n\n"+escaped_markdown_content + """\n\n\n""" +"""
    
        Issue:
        {issue}
    
        Please provide Root cause analysis of the issue. Any Specific code fix based on the code given in the document.Also any potential side effects or considerations
    
     DO NOT USE ANY EXTERNAL KNOWLEDGE OR INFORMATION. STRICTLY USE THE DOCUMENT PROVIDED ABOVE."""
    
    
    # Combine the initial prompt with the escaped markdown content
    final_pr = initial_pr 
    
    
    # Now create the ChatPromptTemplate
    # No need to convert to str(final_pr) if final_pr is already a string
    prompt = ChatPromptTemplate.from_template(final_pr)
    
    # You can then use this prompt, for example:
    # formatted_prompt = prompt.format_prompt(issue="Describe your actual issue here")
    # print(formatted_prompt.to_string())
    chain = (
    {"issue": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser())

    embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest")

    # === Step 6: Embed All Issues for Similarity Analysis ===
    try:
        embedded_issues = embedding_model.embed_documents(issues)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("Proceeding without duplicate detection...")
        embedded_issues = None

    duplicates = set()
    if embedded_issues:
        duplicate_threshold = 0.85  # Adjust similarity threshold as needed
        num_issues = len(issues)
        
        for i in range(num_issues):
            for j in range(i + 1, num_issues):
                try:
                    sim = cosine_similarity([embedded_issues[i]], [embedded_issues[j]])[0][0]
                    if sim > duplicate_threshold:
                        duplicates.add(j)
                except Exception as e:
                    print(f"Error calculating similarity between issues {i+1} and {j+1}: {e}")

    # === Step 8: Filter Out Duplicates ===
    unique_issues = [issue for idx, issue in enumerate(issues) if idx not in duplicates]
    analyzed_issues = []
    responses = []
    failed_analyses = []
    # === Step 9: Analyze Each Unique Issue with codellama ===
    
    for idx, issue_text in enumerate(unique_issues):
        original_idx = [i for i, issue in enumerate(issues) if issue == issue_text][0]
        
        
        try:
            # Invoke the codellama analysis
            response = chain.invoke(input=issue_text)
            analyzed_issues.append(issue_text)
            responses.append(response)
            
        except Exception as e:
            failed_analyses.append({
                'issue_idx': original_idx + 1,
                'issue_text': issue_text,
                'error': str(e)
            })
            continue
        
    return str(responses[0]) if len(responses)>=1 else "No response"


def main():
    if len(sys.argv) != 3:
        print("Usage: python down.py <issue_title> <issue_body>")
        sys.exit(1)
        
    with open('./issue_title.txt', 'r') as f:
        title = f.read()

    with open('./issue_body.txt', 'r') as f:
        body = f.read()

    response=callCodeLLama(title,body)
    result = (
        "### 🔍 Issue Information Extracted by `down.py`:\n\n"
        f"**Title:** {title}\n\n"
        f"**Body:** {body}\n"
        f"**Response:** {response}\n"
    )
    print(result)

if __name__ == "__main__":
    main()
