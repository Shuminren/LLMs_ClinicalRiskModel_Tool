#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main program for clinical risk prediction model information extraction
Fetches article information from PubMed and PMC, and uses LLM to extract structured data
"""

# =====================================
# 1. Model Configuration
# =====================================
API_KEY = "your-openai-api-key-here"
MODEL = "grok-3"

# =====================================
# 2. Model Configuration and Utility Functions
# =====================================
import openai
from rich.console import Console
import time
from openai import OpenAIError  

console = Console()

def chat_completion(messages, max_retries=3, backoff_factor=2):
    """Call LLM API for conversation"""
    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(
                api_key=API_KEY,
                base_url="https://api.x.ai/v1"
            )
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except OpenAIError as e:  
            if e.status_code == 429:  
                sleep_time = backoff_factor ** attempt  
                console.print(f"[bold yellow]Rate limit hit (429). Retrying after {sleep_time} seconds... (Attempt {attempt+1}/{max_retries})[/]")
                time.sleep(sleep_time)
            else:
                console.print(f"[bold red]API request failed: {str(e)}[/]")
                return None
    console.print(f"[bold red]Max retries reached for API call.[/]")
    return None

def read_pmid_from_txt(file_path):
    """Read PMID list from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            pmids = [line.strip() for line in file if line.strip()]
        return pmids
    except FileNotFoundError:
        console.print(f"[bold red]Error: File {file_path} not found.[/]")
        return []
    except Exception as e:
        console.print(f"[bold red]Error reading {file_path}: {str(e)}[/]")
        return []

# =====================================
# 3. PubMed Metadata Scraping Module
# =====================================
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_driver():
    """Configure Selenium WebDriver for PubMed page scraping"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124")
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def fetch_pubmed_data(pmid, driver):
    """
    Scrape article metadata from PubMed page
    
    Args:
        pmid: PubMed article ID
        driver: Selenium WebDriver instance
    
    Returns:
        dict: Metadata including title, authors, DOI, keywords, journal name, PMCID, etc.
    """
    url = f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
    driver.get(url)
    
    data = {'pmid': pmid}
    wait = WebDriverWait(driver, 5)
    
    try:
        data['title'] = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'heading-title'))).text.strip()
    except:
        data['title'] = None

    try:
        authors = driver.find_element(By.CLASS_NAME, 'authors-list').text
        data['authors'] = ', '.join(re.sub(r'\s*\d+', '', authors).split(','))
    except:
        data['authors'] = None

    try:
        affiliation = driver.find_element(By.CLASS_NAME, 'affiliation-link').get_attribute('title')
        data['first_author_last_affiliation_word'] = affiliation.split(',')[-1].strip().split()[-1].rstrip('.')
    except:
        data['first_author_last_affiliation_word'] = None

    try:
        data['doi'] = driver.find_element(By.XPATH, '//a[@data-ga-action="DOI"]').text.strip()
    except:
        data['doi'] = None

    try:
        keywords = driver.find_element(By.XPATH, '//p[strong[contains(text(),"Keywords")]]').text
        data['keywords'] = keywords.replace("Keywords:", "").strip().rstrip('.')
    except:
        data['keywords'] = None

    try:
        data['journal_name'] = driver.find_element(By.XPATH, '//meta[@name="citation_publisher"]').get_attribute('content').strip()
    except:
        data['journal_name'] = None

    try:
        pmcid_element = driver.find_element(By.XPATH, '//a[contains(@href, "pmc.ncbi.nlm.nih.gov/articles/PMC")]')
        pmcid_full = pmcid_element.text.strip()
        data['pmcid'] = pmcid_full.replace("PMCID: ", "").strip()
    except:
        data['pmcid'] = None

    return data

# =====================================
# 4. PMC Full Text Scraping Module (v2.1)
# =====================================
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum
import traceback

# Optional: Selenium support
try:
    from selenium.webdriver.chrome.options import Options as SeleniumOptions
    from selenium.webdriver.common.by import By as SeleniumBy
    from selenium.webdriver.support.ui import WebDriverWait as SeleniumWebDriverWait
    from selenium.webdriver.support import expected_conditions as SeleniumEC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠ Selenium not installed, will only use requests for scraping")

# Status and result classes
class ExtractionStatus(Enum):
    """Extraction status enumeration"""
    SUCCESS = "success"
    PARTIAL = "partial"      # Partially successful (used fallback)
    FAILED = "failed"
    NO_PMCID = "no_pmcid"

@dataclass
class ExtractionResult:
    """Extraction result data class"""
    pmcid: str
    status: ExtractionStatus
    full_text: Optional[str] = None
    full_text_chunks: Optional[List[str]] = None
    error_message: Optional[str] = None
    method_used: Optional[str] = None
    sections_found: Optional[List[str]] = None
    char_count: int = 0
    word_count: int = 0

# Configuration constants
EXCLUDE_SECTION_PATTERNS = [
    r'^references?$',
    r'^bibliography$',
    r'^acknowledgm?ents?$',
    r'^acknowledg?ments?$',
    r'^author\s*contributions?$',
    r'^authors?\s*contributions?$',
    r'^contributors?$',
    r'^contributor\s*information$',
    r'^conflicts?\s*of\s*interests?$',
    r'^competing\s*interests?$',
    r'^declarations?$',
    r'^disclosure$',
    r'^funding$',
    r'^funding\s*(sources?|information)?$',
    r'^financial\s*disclosure$',
    r'^supplementary\s*(materials?|data|information)?$',
    r'^supporting\s*information$',
    r'^appendi(x|ces)$',
    r'^data\s*availability',
    r'^ethics\s*(statement|approval)?$',
    r'^ethical\s*approval$',
    r'^footnotes?$',
    r'^abbreviations?$',
    r'^associated\s*data$',
]

MAIN_CONTENT_PATTERNS = [
    r'^abstract$',
    r'^highlights?$',
    r'^background$',
    r'^introduction$',
    r'^methods?$',
    r'^materials?\s*(and|&)\s*methods?$',
    r'^patients?\s*(and|&)\s*methods?$',
    r'^study\s*design$',
    r'^results?$',
    r'^findings?$',
    r'^discussion$',
    r'^conclusions?$',
    r'^summary$',
]

# Utility functions
def is_exclude_section(title: str) -> bool:
    """Check if section title should be excluded"""
    if not title:
        return False
    title_clean = title.strip().lower()
    for pattern in EXCLUDE_SECTION_PATTERNS:
        if re.match(pattern, title_clean, re.IGNORECASE):
            return True
    return False

def is_main_content_section(title: str) -> bool:
    """Check if section title is a main content section"""
    if not title:
        return False
    title_clean = title.strip().lower()
    for pattern in MAIN_CONTENT_PATTERNS:
        if re.match(pattern, title_clean, re.IGNORECASE):
            return True
    return False

def clean_text_v2(text: str) -> str:
    """Clean text: remove excessive whitespace, ORCID, emails, etc."""
    if not text:
        return ""
    
    # Remove ORCID ID
    text = re.sub(r'ORCID\s*(ID)?\s*:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://orcid\.org/\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove duplicate content in Figure/Table annotations
    text = re.sub(r'(Figure|Table)\s*\d+\.?\s*(Open in a new tab)?', r'\1 ', text, flags=re.I)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def chunk_text_v2(text: str, chunk_size_words: int = 4000, overlap_words: int = 1000) -> List[str]:
    """Chunk text with overlap"""
    words = text.split()
    if len(words) <= chunk_size_words:
        return [text]
    
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size_words - overlap_words
    return chunks

# PMC scraper class
class PMCScraperV2:
    """PMC full text scraper v2.1"""
    
    def __init__(self, use_selenium: bool = False, timeout: int = 30):
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.timeout = timeout
        self.driver = None
        self.request_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
    def setup_driver(self):
        """Configure Selenium WebDriver"""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not installed")
        chrome_options = SeleniumOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument(f"user-agent={self.request_headers['User-Agent']}")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        self.driver = webdriver.Chrome(options=chrome_options)
        return self.driver
    
    def close_driver(self):
        """Close WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
    
    def fetch_html(self, pmcid: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch HTML content from PMC page"""
        url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
        
        # Method 1: Use requests (faster and more stable)
        try:
            response = requests.get(url, headers=self.request_headers, timeout=self.timeout)
            response.raise_for_status()
            
            if len(response.text) > 10000:
                console.print(f"[green]  ✓ requests succeeded ({len(response.text)//1024}KB)[/]")
                return response.text, None
            else:
                console.print(f"[yellow]  ⚠ Page content too short ({len(response.text)}B)[/]")
        except requests.Timeout:
            console.print(f"[yellow]  ⚠ requests timeout, trying Selenium...[/]")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None, f"Article not found (404): {pmcid}"
            console.print(f"[yellow]  ⚠ HTTP error {e.response.status_code}[/]")
        except requests.RequestException as e:
            console.print(f"[yellow]  ⚠ requests failed: {str(e)[:50]}[/]")
        
        # Method 2: Selenium fallback
        if self.use_selenium:
            try:
                console.print("[cyan]  Attempting Selenium scraping...[/]")
                if not self.driver:
                    self.setup_driver()
                self.driver.get(url)
                SeleniumWebDriverWait(self.driver, self.timeout).until(
                    SeleniumEC.presence_of_element_located((SeleniumBy.CLASS_NAME, 'main-article-body'))
                )
                time.sleep(1)
                html = self.driver.page_source
                if len(html) > 10000:
                    console.print(f"[green]  ✓ Selenium succeeded ({len(html)//1024}KB)[/]")
                    return html, None
            except Exception as e:
                console.print(f"[red]  ✗ Selenium failed: {str(e)[:50]}[/]")
                return None, f"Selenium failed: {str(e)}"
        
        return None, "Unable to fetch page content"
    
    def extract_method1_main_body(self, soup: BeautifulSoup) -> Tuple[Optional[str], List[str]]:
        """Method 1: Precise extraction based on main-article-body (recommended)"""
        sections_found = []
        content_parts = []
        
        main_body = soup.find('section', class_='main-article-body')
        if not main_body:
            main_body = soup.find('div', class_='article-content') or \
                        soup.find('article', class_='article')
        
        if not main_body:
            return None, ["main-article-body not found"]
        
        console.print(f"[dim]    Found main article container[/]")
        
        for section in main_body.find_all('section', recursive=False):
            heading = section.find(['h2', 'h3', 'h4'])
            section_title = heading.get_text(strip=True) if heading else ""
            
            if section_title and is_exclude_section(section_title):
                console.print(f"[dim]    ⏹ Stopped at: {section_title}[/]")
                break
            
            section_text = section.get_text(separator=' ', strip=True)
            if section_text and len(section_text) > 50:
                content_parts.append(section_text)
                if section_title:
                    sections_found.append(section_title)
                    console.print(f"[dim]    ✓ Extracted: {section_title[:40]}... ({len(section_text)} chars)[/]")
        
        if content_parts:
            return ' '.join(content_parts), sections_found
        return None, sections_found
    
    def extract_method2_heading(self, soup: BeautifulSoup) -> Tuple[Optional[str], List[str]]:
        """Method 2: Extraction based on heading tag positioning"""
        sections_found = []
        content_parts = []
        headings = soup.find_all(['h2', 'h3'])
        start_found = False
        
        for heading in headings:
            heading_text = heading.get_text(strip=True)
            
            if is_exclude_section(heading_text):
                console.print(f"[dim]    ⏹ Stopped at: {heading_text}[/]")
                break
            
            if is_main_content_section(heading_text):
                start_found = True
            
            if start_found:
                sections_found.append(heading_text)
                
                content = []
                current = heading.find_next_sibling()
                while current:
                    if current.name in ['h2', 'h3']:
                        break
                    text = current.get_text(separator=' ', strip=True)
                    if text:
                        content.append(text)
                    current = current.find_next_sibling()
                
                if content:
                    content_parts.append(f"{heading_text}\n{' '.join(content)}")
        
        if content_parts:
            return '\n\n'.join(content_parts), sections_found
        return None, sections_found
    
    def extract_method3_fallback(self, soup: BeautifulSoup) -> Tuple[Optional[str], List[str]]:
        """Method 3: Fallback - Extract all then truncate using regex"""
        sections_found = ["[Fallback mode]"]
        
        article = soup.find('article') or \
                  soup.find('div', class_='article-content') or \
                  soup.find('div', id=re.compile(r'article|content', re.I)) or \
                  soup.find('main')
        
        if not article:
            return None, sections_found
        
        for tag in article.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        for heading in article.find_all(['h1', 'h2', 'h3', 'h4']):
            if is_exclude_section(heading.get_text(strip=True)):
                for sibling in list(heading.find_next_siblings()):
                    sibling.decompose()
                heading.decompose()
                break
        
        text = article.get_text(separator=' ', strip=True)
        
        for pattern in [r'\bReferences\b', r'\bBibliography\b', r'\bFootnotes\b']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                text = text[:match.start()]
                break
        
        return (text.strip(), sections_found) if len(text) > 500 else (None, sections_found)
    
    def extract_full_text(self, pmcid: str) -> ExtractionResult:
        """Main extraction function: Try multiple methods sequentially to extract full text"""
        if not pmcid:
            return ExtractionResult(
                pmcid="", 
                status=ExtractionStatus.NO_PMCID, 
                error_message="PMCID is empty"
            )
        
        pmcid = pmcid.strip()
        if not pmcid.upper().startswith('PMC'):
            pmcid = f"PMC{pmcid}"
        pmcid = pmcid.upper()
        
        console.print(f"\n[bold blue]📄 Fetching {pmcid}...[/]")
        
        try:
            html, error = self.fetch_html(pmcid)
            if not html:
                return ExtractionResult(
                    pmcid=pmcid, 
                    status=ExtractionStatus.FAILED, 
                    error_message=error or "Unable to fetch HTML"
                )
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Method 1: Precise extraction using main-article-body
            console.print("[cyan]  Method 1: main-article-body precise extraction[/]")
            text, sections = self.extract_method1_main_body(soup)
            if text and len(text) > 1000:
                clean = clean_text_v2(text)
                console.print(f"[green]  ✓ Method 1 succeeded ({len(sections)} sections, {len(clean)} chars)[/]")
                return ExtractionResult(
                    pmcid=pmcid, 
                    status=ExtractionStatus.SUCCESS,
                    full_text=clean, 
                    full_text_chunks=chunk_text_v2(clean),
                    method_used="method1_main_body", 
                    sections_found=sections,
                    char_count=len(clean),
                    word_count=len(clean.split())
                )
            
            # Method 2: Heading-based extraction
            console.print("[cyan]  Method 2: Heading-based extraction[/]")
            text, sections = self.extract_method2_heading(soup)
            if text and len(text) > 1000:
                clean = clean_text_v2(text)
                console.print(f"[green]  ✓ Method 2 succeeded ({len(sections)} sections, {len(clean)} chars)[/]")
                return ExtractionResult(
                    pmcid=pmcid, 
                    status=ExtractionStatus.SUCCESS,
                    full_text=clean, 
                    full_text_chunks=chunk_text_v2(clean),
                    method_used="method2_heading", 
                    sections_found=sections,
                    char_count=len(clean),
                    word_count=len(clean.split())
                )
            
            # Method 3: Fallback
            console.print("[yellow]  Method 3: Fallback extraction[/]")
            text, sections = self.extract_method3_fallback(soup)
            if text and len(text) > 500:
                clean = clean_text_v2(text)
                console.print(f"[yellow]  ⚠ Fallback succeeded ({len(clean)} chars)[/]")
                return ExtractionResult(
                    pmcid=pmcid, 
                    status=ExtractionStatus.PARTIAL,
                    full_text=clean, 
                    full_text_chunks=chunk_text_v2(clean),
                    method_used="method3_fallback", 
                    sections_found=sections,
                    char_count=len(clean),
                    word_count=len(clean.split())
                )
            
            return ExtractionResult(
                pmcid=pmcid, 
                status=ExtractionStatus.FAILED, 
                error_message="All extraction methods failed, page structure may be non-standard"
            )
            
        except Exception as e:
            console.print(f"[red]  ✗ Exception: {str(e)}[/]")
            return ExtractionResult(
                pmcid=pmcid, 
                status=ExtractionStatus.FAILED, 
                error_message=f"Exception: {str(e)}\n{traceback.format_exc()[:500]}"
            )

def fetch_pmc_full_text_v2(pmcid: str, driver=None) -> Dict:
    """Convenience function: Fetch PMC full text (compatible with original interface)"""
    scraper = PMCScraperV2(use_selenium=False)
    result = scraper.extract_full_text(pmcid)
    
    return {
        'pmcid': result.pmcid,
        'full_text': result.full_text,
        'full_text_chunks': result.full_text_chunks,
        'status': result.status.value,
        'error': result.error_message,
        'method': result.method_used,
        'sections': result.sections_found,
        'char_count': result.char_count,
        'word_count': result.word_count
    }

# =====================================
# 5. Prompt Definitions
# =====================================
def get_prompts():
    """Define prompts for information extraction"""
    prompt1 = """
You are a medical research assistant specializing in the study of clinical risk prediction models. You assist users in collecting information and providing answers by identifying and analyzing medical information, statistical data, and prediction models within text. 
[Extracted text]
Please analyze the above text and extract detailed information related to the study design and its key parameters. Specifically, extract the following elements.
1. Study Type: Indicate whether the data used in the research comes from a prospective study (cohort study) or a retrospective study (case-control study).
2. Disease Name: Specify the target disease studied in the text.
3. Data Sources: Extract details about the primary data source, such as the name of the dataset, or the specific institution or public database from which the data were obtained.
4. External Validation: Determine whether the study includes an external validation set, which refers to a dataset originating from a different source than the model development set. All subsequent references to the validation set refer specifically to external validation, not internal test sets.
5. Date Range: Specify the dates of the collected participant data for model development and model external validation, including the start and end dates.
6. Follow-up Time: Specify the follow-up time of the cohort for the model development stage, if applicable.
7. Sample Characteristics: Indicate whether the included samples had a specific disease history, underwent particular surgeries or treatments, or were in a certain condition (e.g., pregnancy, smoking).
8. Sample Sizes: Extract the sample sizes for the development set (including training set and test set) and external validation set, if mentioned.
9. Case and Control Numbers: What are the numbers of cases and controls in the development set? What are the numbers of cases and controls in the external validation set? Perform inference and calculation as much as possible based on the provided sample data.
10. Participant Statistics: Extract demographic information about participants in the development and external validation stages, including: the number of female participants; the number of male participants. Perform inference and calculation as much as possible based on the provided sample data.
11. Extract age information about participants in the development and external validation stages, including the age range of participants; the average age and standard deviation of participants; the median age and interquartile range of participants.
12. The racial or ethnic composition of participants in the development and external validation stages.
    """
    prompt2 = """
Continue extracting the following details related to the prediction models from the research. If the paper constructs or evaluates multiple models, please extract the following information for all models.
1. Model Type: Identify the type of prediction models used (e.g., logistic regression, random forests, deep learning).
2. Prediction Variables: List the variables used in every prediction model.
3. Report the AUC values, including confidence intervals, for the development stage and the validation stage.
4. Report the C-index values, including confidence intervals, for the development stage and the validation stage.
5. Report accuracy values, including confidence intervals, for the development stage and the validation stage.
6. Report F1 scores, including confidence intervals, for the development stage and the validation stage.
7. Determine whether calibration values are reported for the development stage and the validation stage, such as calibration curve, Hosmer–Lemeshow test, Brier Score, or expected vs. observed outcomes. If available, please report them.
8. Nomogram Development: Indicate whether a nomogram was constructed based on the model.
9. Use of TRIPOD Guidelines: State whether the study followed the TRIPOD guidelines, if applicable.
    """
    prompt3 = """
Please format the extracted information into two columns: one column for the field and one column for the value, ensuring that each field corresponds to only one value. Use the following fields to create the table. This table consists of 25 fields.
1. Study Type: e.g., prospective study, retrospective study. If it is not mentioned in the text, answer NA.
2. Disease Name: e.g. lung cancer.
3. External Validation: Answer Yes or No.
4. Date Range in the Development Set: e.g., 1992–2008. If it is not mentioned in the text, answer NA.
5. Date Range in the Validation Set: e.g., 1992–2008. If this study lacks external validation, answer N/A.
6. Median Follow-up Time (Years) and IQR: e.g., 2.95 years (IQR: 1.71–4.83). If it is not mentioned in the text, answer NA.
7. Mean Follow-up Time (Years) and Standard Error: e.g., 2.95 years (SD: 0.71). If it is not mentioned in the text, answer NA.
8. Data Sources: e.g., the Kaiser Permanente Washington Breast Cancer Surveillance Consortium (BCSC) registry, the GEO database, the Cancer Screening Program in Urban China (CanSPUC) conducted in Henan Province. If it is not mentioned in the text, answer NA.
9. Sample Characteristics: e.g., patients with a history of diabetes, pregnant women, ever-smokers. If it is not mentioned in the text, answer N/A.
10. Number of Cases in the Development Set: e.g., 5,323. If it is not mentioned in the text, answer NA.
11. Number of Controls in the Development Set: e.g., 5,323. If it is not mentioned in the text, answer NA.
12. Number of Cases in the External Validation Set: e.g., 5,323. If this study lacks external validation, answer N/A.
13. Number of Controls in the External Validation Set: e.g., 5,323. If this study lacks external validation, answer N/A.
14. Number of Female Participants (Development): e.g., 86. If it is not mentioned in the text, answer NA.
15. Number of Female Participants (External Validation): e.g., 86. If this study lacks external validation, answer N/A.
16. Number of Male Participants (Development): e.g., 545. If it is not mentioned in the text, answer NA.
17. Number of Male Participants (External Validation): e.g., 545. If this study lacks external validation, answer N/A.
18. Age Range (Development): e.g., 22–60. If it is not mentioned in the text, answer NA.
19. Age Range (External Validation): e.g., 22–60. If this study lacks external validation, answer N/A.
20. Average Age and Standard Deviation (Development): e.g., 74.2 years (SD = 8.2). If it is not mentioned in the text, answer NA.
21. Average Age and Standard Deviation (External Validation): e.g., 74.2 years (SD = 8.2). If this study lacks external validation, answer N/A.
22. Median Age (Development) and IQR: e.g., 43.2 years (IQR: 32.43–64.83). If it is not mentioned in the text, answer NA.
23. Median Age (External Validation) and IQR: e.g., 43.2 years (IQR: 32.43–64.83). If this study lacks external validation, answer N/A.
24. Racial/Ethnic Composition (Development): e.g., White: 80.4%, Asian: 8.8%, Black: 3.9%, Other or multiple races: 5.3%, Unknown: 1.5%. If it is not mentioned in the text, answer NA.
25. Racial/Ethnic Composition (External Validation): e.g., White: 80.4%, Asian: 8.8%, Black: 3.9%, Other or multiple races: 5.3%, Unknown: 1.5%. If this study lacks external validation, answer N/A.
    """
    prompt4 = """
Please extract and format the following information for the models in the study. Organize the data into a two-column or multi-column table, with one column for the field name (e.g., Model Name, Model Type, etc.) and the remaining columns for values. Each column corresponds to one model's values. If there are multiple models, a multi-column table is needed. Each field should occupy only one row in the table. This table consists of 10 fields.
1. Model Name: e.g., Tyrer-Cuzick Model, LiFeCRC Score.
2. Model Type: e.g., logistic regression, random forest, support vector machine.
3. Prediction Variables: e.g., age, gender, education, smoking status, blood pressure medication, prevalent stroke.
4. AUC Values: e.g., 0.767 (95% CI: 0.749–0.786). If a validation stage is present, prioritize reporting the values from the validation stage.
5. C-index Values: e.g., 0.767 (95% CI: 0.749–0.786). If a validation stage is present, prioritize reporting the values from the validation stage.
6. Accuracy: e.g., 74.8% (95% CI: 71.30% - 78.30%). If a validation stage is present, prioritize reporting the values from the validation stage.
7. F1-score: e.g., 74.8% (95% CI: 71.30% - 78.30%). If a validation stage is present, prioritize reporting the values from the validation stage.
8. Calibration Values: e.g., Hosmer-Lemeshow (HL) test p-value = 0.428; O/E ratio ranged from 0.79 to 1.22, or a brief description. If a validation stage is present, prioritize reporting the values from the validation stage.
9. Nomogram Application: Answer Yes or No.
10. Use of TRIPOD Guidelines: Answer Yes or No.
    """
    return [prompt1, prompt2, prompt3, prompt4]

# =====================================
# 6. Structured Data Storage
# =====================================
import csv

# Define fields
sample_fields = [
    "PMID",  # Add PMID field to match extraction function
    "Study Type",
    "Disease Name",
    "External Validation",
    "Date Range in the Development Set",
    "Date Range in the Validation Set",
    "Median Follow-up Time (Years) and IQR",
    "Mean Follow-up Time (Years) and Standard Error",
    "Data Sources",
    "Sample Characteristics",
    "Number of Cases in the Development Set",
    "Number of Controls in the Development Set",
    "Number of Cases in the Validation Set",
    "Number of Controls in the Validation Set",
    "Number of Female Participants (Development)",
    "Number of Female Participants (Validation)",
    "Number of Male Participants (Development)",
    "Number of Male Participants (Validation)",
    "Age Range (Development)",
    "Age Range (Validation)",
    "Average Age and Standard Deviation (Development)",
    "Average Age and Standard Deviation (Validation)",
    "Median Age (Development) and IQR",
    "Median Age (Validation) and IQR",
    "Racial/Ethnic Composition (Development)",
    "Racial/Ethnic Composition (Validation)"
]

model_fields = [
    "PMID",  # Add PMID field to match extraction function
    "Model Name",
    "Model Type",
    "Prediction Variables",
    "AUC Values",
    "C-index Values",
    "Accuracy",
    "F1-score",
    "Calibration Values",
    "Nomogram Application",
    "Use of TRIPOD Guidelines"
]

literature_fields = [
    "PMID",
    "Title",
    "Authors",
    "First Author Last Affiliation Word",
    "DOI",
    "Keywords",
    "Journal Name",
    "PMCID",
    "Full Text"
]

def process_input(input_text, prompts):
    """Process input text and get LLM responses using prompts"""
    responses = []
    for i, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": prompt.replace("[Extracted text]", input_text)}]
        console.print(f"[italic yellow]Processing Prompt {i}...[/]")
        response = chat_completion(messages)
        if response:
            responses.append(response)
            console.print(f"\n[bold magenta]Response {i}:[/]")
            console.print(response)
            console.print("\n" + "-"*50 + "\n")
        else:
            console.print(f"[bold red]Prompt {i} processing failed[/]")
    return responses

def extract_sample_data(response_sample, pmid):
    """Extract sample data from response"""
    data_dict = {field: "" for field in sample_fields}
    data_dict["PMID"] = pmid
    lines_sample = response_sample.splitlines()
    
    for line in lines_sample:
        match = re.match(r'\s*\d+\.\s+([^0-9].*?)\s{2,}(.*)', line.strip())
        if match:
            field, value = match.groups()
            for sample_field in sample_fields:
                if sample_field in field:
                    data_dict[sample_field] = value
                    break

    return data_dict

def extract_model_data(response_model, pmid):
    """Extract model data from response, handling multiple models"""
    model_data_list = []
    lines_model = response_model.splitlines()
    
    current_model = {field: "" for field in model_fields}
    current_model["PMID"] = pmid
    
    for line in lines_model:
        match = re.match(r'\s*\d+\.\s+([^0-9].*?)\s{2,}(.*)', line.strip())
        if match:
            field, value = match.groups()
            if "Model Name" in field and current_model["Model Name"]:
                # Detected new model, save current model and start new one
                model_data_list.append(current_model)
                current_model = {field: "" for field in model_fields}
                current_model["PMID"] = pmid
            for model_field in model_fields:
                if model_field in field:
                    current_model[model_field] = value
                    break
    
    # Append the last model
    if current_model["Model Name"] or any(current_model[field] for field in model_fields if field != "PMID"):
        model_data_list.append(current_model)
    
    # If no models were identified, return an empty model record
    if not model_data_list:
        model_data_list.append({field: "" for field in model_fields})
        model_data_list[0]["PMID"] = pmid
    
    return model_data_list

# =====================================
# 7. Main Program
# =====================================
def main():
    """Main function"""
    console.print("[bold green]Welcome to the Clinical Risk Prediction Model Analysis Tool[/]")
    txt_file = 'PMID.TXT'
    pmids = read_pmid_from_txt(txt_file)
    if not pmids:
        console.print("[bold red]Failed to read PMID[/]")
        return

    with open('literature_data.csv', 'a', newline='', encoding='utf-8') as literature_csv, \
         open('sample_data.csv', 'a', newline='', encoding='utf-8') as sample_csv, \
         open('model_information.csv', 'a', newline='', encoding='utf-8') as model_csv:
        
        literature_writer = csv.DictWriter(literature_csv, fieldnames=literature_fields)
        sample_writer = csv.DictWriter(sample_csv, fieldnames=sample_fields)
        model_writer = csv.DictWriter(model_csv, fieldnames=model_fields)
        
        if literature_csv.tell() == 0:
            literature_writer.writeheader()
        if sample_csv.tell() == 0:
            sample_writer.writeheader()
        if model_csv.tell() == 0:
            model_writer.writeheader()

        total_pmids = len(pmids)
        success_count = 0
        fail_count = 0
        
        for idx, pmid in enumerate(pmids, 1):
            driver = None
            try:
                console.print(f"\n[bold cyan][{idx}/{total_pmids}] Processing PMID: {pmid}...[/]")
                driver = setup_driver()
                pubmed_data = fetch_pubmed_data(pmid, driver)
                
                # Use optimized v2.1 version to fetch PMC full text
                if pubmed_data.get('pmcid'):
                    full_text_data = fetch_pmc_full_text_v2(pubmed_data['pmcid'], driver)
                    full_text = full_text_data.get('full_text', "")
                else:
                    console.print(f"[yellow]No PMCID found for PMID {pmid}, skipping full text extraction.[/]")
                    full_text = ""
                
                # Write literature data
                literature_writer.writerow({
                    'PMID': pmid,
                    'Title': pubmed_data.get('title', ''),
                    'Authors': pubmed_data.get('authors', ''),
                    'First Author Last Affiliation Word': pubmed_data.get('first_author_last_affiliation_word', ''),
                    'DOI': pubmed_data.get('doi', ''),
                    'Keywords': pubmed_data.get('keywords', ''),
                    'Journal Name': pubmed_data.get('journal_name', ''),
                    'PMCID': pubmed_data.get('pmcid', ''),
                    'Full Text': full_text
                })
                literature_csv.flush()  # Flush to disk immediately
                
                # If full text is available, perform information extraction
                if full_text:
                    prompts = get_prompts()
                    prompt1_with_fulltext = prompts[0].replace("[Extracted text]", full_text)
                    responses = process_input(prompt1_with_fulltext, prompts)
                    
                    if len(responses) >= 4:
                        response_sample = responses[2]  # Prompt 3
                        response_model = responses[3]  # Prompt 4
                        
                        sample_data = extract_sample_data(response_sample, pmid)
                        model_data_list = extract_model_data(response_model, pmid)
                        
                        sample_writer.writerow(sample_data)
                        sample_csv.flush()  # Flush to disk immediately
                        
                        # Handle multiple models
                        for model_data in model_data_list:
                            model_writer.writerow(model_data)
                        model_csv.flush()  # Flush to disk immediately
                    else:
                        console.print(f"[yellow]Warning: Only {len(responses)} responses received for PMID {pmid}, expected 4.[/]")
                        # Even if response is incomplete, write empty data for consistency
                        sample_data = {field: "" for field in sample_fields}
                        sample_data["PMID"] = pmid
                        sample_writer.writerow(sample_data)
                        sample_csv.flush()
                        
                        model_data = {field: "" for field in model_fields}
                        model_data["PMID"] = pmid
                        model_writer.writerow(model_data)
                        model_csv.flush()
                else:
                    console.print(f"[yellow]No full text available for PMID {pmid}, skipping information extraction.[/]")
                    # Even if no full text, write empty data for consistency
                    sample_data = {field: "" for field in sample_fields}
                    sample_data["PMID"] = pmid
                    sample_writer.writerow(sample_data)
                    sample_csv.flush()
                    
                    model_data = {field: "" for field in model_fields}
                    model_data["PMID"] = pmid
                    model_writer.writerow(model_data)
                    model_csv.flush()
                
                success_count += 1
                console.print(f"[green]✓ Information related to PMID {pmid} has been saved. ({success_count} success, {fail_count} failed)[/]")
                
            except Exception as e:
                fail_count += 1
                console.print(f"[bold red]✗ Error processing PMID {pmid}: {str(e)}[/]")
                console.print(f"[dim]{traceback.format_exc()[:200]}[/]")
                
                # Even if error occurs, try to write basic information
                try:
                    literature_writer.writerow({
                        'PMID': pmid,
                        'Title': '',
                        'Authors': '',
                        'First Author Last Affiliation Word': '',
                        'DOI': '',
                        'Keywords': '',
                        'Journal Name': '',
                        'PMCID': '',
                        'Full Text': f'[ERROR: {str(e)[:100]}]'
                    })
                    literature_csv.flush()
                    
                    sample_data = {field: "" for field in sample_fields}
                    sample_data["PMID"] = pmid
                    sample_writer.writerow(sample_data)
                    sample_csv.flush()
                    
                    model_data = {field: "" for field in model_fields}
                    model_data["PMID"] = pmid
                    model_writer.writerow(model_data)
                    model_csv.flush()
                except Exception as write_error:
                    console.print(f"[bold red]Failed to write error record: {str(write_error)}[/]")
                
            finally:
                # Ensure driver is always closed
                if driver:
                    try:
                        driver.quit()
                    except Exception as e:
                        console.print(f"[yellow]Warning: Error closing driver for PMID {pmid}: {str(e)}[/]")
        
        # Summary after loop completion
        console.print(f"\n[bold green]Processing completed![/]")
        console.print(f"[green]Total: {total_pmids}, Success: {success_count}, Failed: {fail_count}[/]")

if __name__ == "__main__":
    main()
