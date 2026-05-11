#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Double-LLM-Validation Module
Used to resolve disagreements between two LLM models
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import xml.etree.ElementTree as ET
import openai
from openai import OpenAIError
import traceback
import re

def get_pmcid_from_pmid(pmid):
    """
    Get PMCID from PMID
    
    Args:
        pmid: PubMed article ID
    
    Returns:
        str: PMCID (with PMC prefix), returns None if not found
    """
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&rettype=xml"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        # Try multiple XPath paths to find PMCID
        pmcid_elem = root.find('.//ArticleId[@IdType="pmc"]')
        if pmcid_elem is None:
            pmcid_elem = root.find('.//ArticleIdList/ArticleId[@IdType="pmc"]')
        if pmcid_elem is None:
            # Try to find all ArticleId nodes
            for article_id in root.findall('.//ArticleId'):
                if article_id.get('IdType') == 'pmc':
                    pmcid_elem = article_id
                    break
        
        if pmcid_elem is not None:
            pmcid = pmcid_elem.text.strip()
            # Ensure PMCID format is correct (add PMC prefix if missing)
            if pmcid and not pmcid.upper().startswith('PMC'):
                pmcid = f"PMC{pmcid}"
            return pmcid
        else:
            return None
    except Exception as e:
        print(f"Error fetching PMCID for PMID {pmid}: {str(e)}")
        return None

def extract_pmc_content(pmcid):
    """
    Extract full text content from PMCID
    
    Args:
        pmcid: PMC article ID
    
    Returns:
        str: Extracted article content
    """
    try:
        # Ensure PMCID format is correct
        pmcid_clean = pmcid.strip()
        if not pmcid_clean.upper().startswith('PMC'):
            pmcid_clean = f"PMC{pmcid_clean}"
        
        url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid_clean}/"
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Improved text extraction logic
        content = ""
        
        # Method 1: Try to find content between abstract and references
        abstract_section = soup.find("section", {"class": "abstract", "id": "abstract1"})
        if not abstract_section:
            abstract_section = soup.find("div", {"class": "abstract"})
        if not abstract_section:
            abstract_section = soup.find("section", class_="abstract")
        
        references_section = soup.find("h2", {"class": "pmc_sec_title"}, string="References")
        if not references_section:
            references_section = soup.find("h2", string=re.compile(r"References?", re.I))
        
        if abstract_section and references_section:
            current = abstract_section.find_next()
            while current and current != references_section:
                if hasattr(current, 'name') and current.name == "section":
                    text = current.get_text(separator=" ", strip=True)
                    if text and len(text) > 50:  # Filter content that is too short
                        content += text + "\n"
                current = current.find_next() if hasattr(current, 'find_next') else None
        
        # Method 2: If method 1 fails, try to extract main-article-body
        if not content or len(content) < 500:
            main_body = soup.find('section', class_='main-article-body')
            if main_body:
                content = main_body.get_text(separator=" ", strip=True)
        
        # Method 3: If still fails, extract the entire article tag
        if not content or len(content) < 500:
            article = soup.find('article')
            if article:
                # Remove unwanted parts
                for tag in article.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                content = article.get_text(separator=" ", strip=True)
        
        if not content or len(content) < 100:
            content = "Failed to extract sufficient content from the article."
        
        return content
    except requests.Timeout:
        return f"Timeout error fetching content for PMCID {pmcid}"
    except requests.RequestException as e:
        return f"Request error fetching content for PMCID {pmcid}: {str(e)}"
    except Exception as e:
        return f"Error fetching content for PMCID {pmcid}: {str(e)}"

def disagreement_resolution(openai_key, claude_key, agreement_filepath):
    """
    Resolve disagreements between two LLM models using cross-validation with both GPT-5.2 and Claude
    
    Args:
        openai_key: OpenAI API key for GPT-5.2
        claude_key: Claude API key
        agreement_filepath: Path to Excel file containing disagreement data
            - Must contain columns: "Variable", "GPT5_2_Responses", "CLAUDE_Responses", "Agree(A)/Disagree(D)", "PMID"
    """
    try:
        df = pd.read_excel(agreement_filepath)
    except FileNotFoundError:
        print(f"Error: File '{agreement_filepath}' not found.")
        return
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return
    
    # Check if required columns exist
    required_columns = ["Variable", "GPT5_2_Responses", "CLAUDE_Responses", "Agree(A)/Disagree(D)", "PMID"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Filter out disagreement cases
    df_disagreements = df[df["Agree(A)/Disagree(D)"].astype(str).str.upper().isin(["D", "DISAGREE", "DISAGREEMENT"])].copy()
    
    if len(df_disagreements) == 0:
        print("No disagreements found in the file.")
        return
    
    print(f"Total disagreements found: {len(df_disagreements)}\n")
    
    # Group processing by PMID
    results = []
    processed_pmids = {}  # Cache full text content for processed PMIDs
    
    # Process in order, but grouped by PMID
    for pmid_group, group_df in df_disagreements.groupby("PMID"):
        pmid = str(pmid_group)
        print(f"\n{'='*80}")
        print(f"Processing PMID: {pmid} ({len(group_df)} variables)")
        print(f"{'='*80}")
        
        # Get PMCID and full text (if not yet retrieved)
        if pmid not in processed_pmids:
            pmcid = get_pmcid_from_pmid(pmid)
            if pmcid is None:
                print(f"PMCID not found for PMID {pmid}, skipping all variables for this PMID.")
                # Create error records for all variables
                for idx, row in group_df.iterrows():
                    results.append({
                        "PMID": pmid,
                        "PMCID": "",
                        "Variable": str(row["Variable"]) if pd.notna(row["Variable"]) else "",
                        "GPT5_2_Response": str(row["GPT5_2_Responses"]) if pd.notna(row["GPT5_2_Responses"]) else "",
                        "CLAUDE_Response": str(row["CLAUDE_Responses"]) if pd.notna(row["CLAUDE_Responses"]) else "",
                        "GPT5_2_Verification_of_Claude": "Error: PMCID not found",
                        "Claude_Verification_of_GPT5_2": "Error: PMCID not found"
                    })
                continue
            
            print(f"Fetching full text for PMCID: {pmcid}")
            content = extract_pmc_content(pmcid)
            print(f"Extracted Content Length: {len(content)} characters")
            processed_pmids[pmid] = {"pmcid": pmcid, "content": content}
        else:
            pmcid = processed_pmids[pmid]["pmcid"]
            content = processed_pmids[pmid]["content"]
            print(f"Using cached content for PMID {pmid}")
        
        # Process all variables for this PMID
        for idx, row in group_df.iterrows():
            var = str(row["Variable"]) if pd.notna(row["Variable"]) else ""
            gpt_response = str(row["GPT5_2_Responses"]) if pd.notna(row["GPT5_2_Responses"]) else ""
            claude_response = str(row["CLAUDE_Responses"]) if pd.notna(row["CLAUDE_Responses"]) else ""
            
            print(f"\n  Variable: {var}")
            print(f"  GPT-5.2 Response: {gpt_response[:50]}...")
            print(f"  Claude Response: {claude_response[:50]}...")
            
            # Create two verification prompts: cross-validation
            # GPT-5.2 verifies Claude's response, Claude verifies GPT-5.2's response
            prompt_gpt = f"For the given text, another LLM (Claude) generated the response = [{claude_response}] for the variable = [{var}]. Verify whether Claude's response is correct or incorrect. If the response is incorrect, generate the correct response as short and precisely as possible."
            
            prompt_claude = f"For the given text, another LLM (GPT-5.2) generated the response = [{gpt_response}] for the variable = [{var}]. Verify whether GPT-5.2's response is correct or incorrect. If the response is incorrect, generate the correct response as short and precisely as possible."
            
            # Call GPT-5.2 and Claude for cross-validation
            print("  Calling GPT-5.2 to verify Claude's response...")
            gpt_verification = gpt_function(content, prompt_gpt, openai_key, "gpt-5.2")
            gpt_verification = str(gpt_verification) if gpt_verification else "No response from GPT-5.2"
            
            print("  Calling Claude to verify GPT-5.2's response...")
            claude_verification = claude_function(content, prompt_claude, claude_key, "claude-opus-4-5-20251101")
            claude_verification = str(claude_verification) if claude_verification else "No response from Claude"
            
            # Claude API has TPM limits, add delay
            time.sleep(30)
            
            # Add to results list
            results.append({
                "PMID": pmid,
                "PMCID": pmcid,
                "Variable": var,
                "GPT5_2_Response": gpt_response,
                "CLAUDE_Response": claude_response,
                "GPT5_2_Verification_of_Claude": gpt_verification,
                "Claude_Verification_of_GPT5_2": claude_verification
            })
            
            print(f"  ✓ Completed verification for variable: {var}")
    
    # Save as structured CSV file
    if results:
        output_df = pd.DataFrame(results)
        output_filename = "DisagreementResolution_CrossValidation.csv"
        output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n{'='*80}")
        print(f"Results saved to {output_filename}")
        print(f"Total processed: {len(results)} disagreement cases")
        print(f"Total unique PMIDs: {len(processed_pmids)}")
        print(f"{'='*80}")
    else:
        print("\nNo results to save.")

def claude_function(content, prompt, api_key, model_name):
    """
    Call Claude API for verification
    
    Args:
        content: Text content to verify
        prompt: Verification prompt
        api_key: Claude API key
        model_name: Model name
    
    Returns:
        str: API response content
    """
    try:
        # Note: This needs to be adjusted according to the actual Claude API SDK
        # Below is an example using the anthropic SDK (requires installation: pip install anthropic)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            full_prompt = f"{prompt}\n\nText to verify:\n{content}"
            
            message = client.messages.create(
                    model=model_name if model_name.startswith("claude") else "claude-opus-4-5-20251101",
                max_tokens=2048,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }]
            )
            return message.content[0].text if message.content else "No response"
        except ImportError:
            print("Warning: anthropic package not installed. Please install it: pip install anthropic")
            return "Claude API not available (anthropic package not installed)"
        except Exception as e:
            print(f"Error calling Claude API: {str(e)}")
            return f"Claude API error: {str(e)}"
    except Exception as e:
        return f"Error in claude_function: {str(e)}"

def gpt_function(content, prompt, api_key, model_name):
    """
    Call GPT-5.2 through the official OpenAI API for verification
    
    Args:
        content: Text content to verify
        prompt: Verification prompt
        api_key: OpenAI API key
        model_name: Model name (e.g., "gpt-5.2")
    
    Returns:
        str: API response content
    """
    try:
        full_prompt = f"{prompt}\n\nText to verify:\n{content}"
        
        messages = [{"role": "user", "content": full_prompt}]
        
        for attempt in range(3):  # Maximum 3 retries
            try:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model_name if model_name else "gpt-5.2",
                    messages=messages,
                    temperature=0,
                    max_tokens=2048
                )
                return response.choices[0].message.content
            except OpenAIError as e:
                if e.status_code == 429:  # Rate limit
                    sleep_time = (2 ** attempt) * 5
                    print(f"Rate limit hit. Retrying after {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"API request failed: {str(e)}")
                    return f"GPT-5.2 API error: {str(e)}"
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return f"Error: {str(e)}"
        
        return "Max retries reached for GPT-5.2 API"
    except Exception as e:
        return f"Error in gpt_function: {str(e)}"

def main():
    """Main function"""
    print("="*80)
    print("Double-LLM-Validation Module (Cross-Validation)")
    print("="*80)
    
    a = input("Do you want to execute Disagreement Resolution Module? Y/N: ")
    if a == "Y" or a == "y":
        openai_key = input("Enter the OpenAI API Key for GPT-5.2: ")
        claude_key = input("Enter the Claude API Key: ")
        agreement_filepath = input("Enter the Path to the Annotated Agreement Matching File: ")
        disagreement_resolution(openai_key, claude_key, agreement_filepath)
    else:
        print("You Choose Not to Execute Agreement/Disagreement Module.")

if __name__ == "__main__":
    main()
