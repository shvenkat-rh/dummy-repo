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
           print(markdown_content)
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
    print("Generating embeddings for all issues...")
    try:
        embedded_issues = embedding_model.embed_documents(issues)
        print(f"Successfully generated embeddings for {len(embedded_issues)} issues.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("Proceeding without duplicate detection...")
        embedded_issues = None

    duplicates = set()
    if embedded_issues:
        duplicate_threshold = 0.85  # Adjust similarity threshold as needed
        num_issues = len(issues)
        
        print("Analyzing for duplicate issues...")
        for i in range(num_issues):
            for j in range(i + 1, num_issues):
                try:
                    sim = cosine_similarity([embedded_issues[i]], [embedded_issues[j]])[0][0]
                    if sim > duplicate_threshold:
                        duplicates.add(j)
                        print(f"Potential duplicate found: Issue {i+1} and Issue {j+1} (similarity: {sim:.3f})")
                except Exception as e:
                    print(f"Error calculating similarity between issues {i+1} and {j+1}: {e}")
        
        print(f"Identified {len(duplicates)} likely duplicate issues.")
    else:
        print("Skipping duplicate detection due to embedding issues.")

    # === Step 8: Filter Out Duplicates ===
    unique_issues = [issue for idx, issue in enumerate(issues) if idx not in duplicates]
    print(f"Processing {len(unique_issues)} unique issues (filtered {len(duplicates)} duplicates).")
    analyzed_issues = []
    responses = []
    failed_analyses = []
    # === Step 9: Analyze Each Unique Issue with codellama ===
    print("\n" + "="*60)
    print("STARTING ISSUE ANALYSIS WITH codellama")
    print("="*60)
    
    for idx, issue_text in enumerate(unique_issues):
        original_idx = [i for i, issue in enumerate(issues) if issue == issue_text][0]
        
        print(f"\n{'='*50}")
        print(f"ANALYZING ISSUE {original_idx + 1} of {len(issues)}")
        print(f"{'='*50}")
        
        try:
            # Invoke the codellama analysis
            response = chain.invoke(input=issue_text)
            analyzed_issues.append(issue_text)
            responses.append(response)
            print(f"\n🔍 codellama ANALYSIS:")
            print("-" * 40)
            print(response)
            
        except Exception as e:
            print(f"❌ Error analyzing issue {original_idx + 1}: {e}")
            failed_analyses.append({
                'issue_idx': original_idx + 1,
                'issue_text': issue_text,
                'error': str(e)
            })
            print("Skipping to next issue...")
            continue
        
        print(f"\n{'='*50}")
        print(f"COMPLETED ISSUE {original_idx + 1}")
        print(f"{'='*50}")
    
    print(f"\n🎉 Analysis complete! Processed {len(unique_issues)} unique issues.")
    return str(response)


def main():
    if len(sys.argv) != 3:
        print("Usage: python down.py <issue_title> <issue_body>")
        sys.exit(1)
    
    issue_title = sys.argv[1]
    issue_body = sys.argv[2]
    response=callCodeLLama(issue_title,issue_body)
    result = (
        "### 🔍 Issue Information Extracted by `down.py`:\n\n"
        f"**Title:** {issue_title}\n\n"
        f"**Body:** {issue_body}\n"
        f"**Response:** {response}\n"
    )
    print(result)

if __name__ == "__main__":
    main()
