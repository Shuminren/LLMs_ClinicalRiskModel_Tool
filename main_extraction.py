#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main program for clinical risk prediction model information extraction
Fetches article metadata from PubMed, then obtains full text either by online
PMC retrieval or by local PMID-indexed TXT/PDF import.
"""

# =====================================
# 1. Model Configuration
# =====================================
API_KEY = "your-openai-api-key-here"
MODEL = "claude-opus-4-5-20251101"
API_MAX_INPUT_TOKENS = 180000

# =====================================
# 2. Model Configuration and Utility Functions
# =====================================
from rich.console import Console
import time
import math
from pathlib import Path

console = Console()

def chat_completion(messages, max_retries=3, backoff_factor=2):
    """Call LLM API for conversation"""
    try:
        import anthropic
    except ImportError:
        console.print("[bold red]anthropic package is required. Install with: pip install anthropic[/]")
        return None

    for attempt in range(max_retries):
        try:
            client = anthropic.Anthropic(api_key=API_KEY)
            user_content = "\n\n".join(
                msg.get("content", "") for msg in messages if msg.get("role") == "user"
            ).strip()
            response = client.messages.create(
                model=MODEL,
                max_tokens=6000,
                temperature=0.5,
                messages=[{"role": "user", "content": user_content}]
            )
            return response.content[0].text if response.content else None
        except Exception as e:
            error_text = str(e).lower()
            if "rate limit" in error_text or "429" in error_text:
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


def choose_full_text_source():
    """Ask the user to choose the full-text source mode."""
    console.print("\n[bold cyan]Choose full-text source:[/]")
    console.print("1. Online PMC retrieval")
    console.print("2. Local PMID-named TXT/PDF files")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "pmc"
        if choice == "2":
            return "local"
        console.print("[yellow]Invalid choice. Please enter 1 or 2.[/]")

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

def chunk_text_v2(
    text: str,
    max_input_tokens: Optional[int] = None,
    chunk_size_tokens: int = 4000,
    overlap_tokens: int = 1000,
    token_per_word: float = 1.3,
    overlap_words: Optional[int] = None
) -> List[str]:
    """Chunk text with an approximate token-based overlap.

    The fallback ``overlap_words`` parameter is retained for compatibility with
    older calls, but new calls should use ``overlap_tokens``.
    """
    words = text.split()
    if not words:
        return []

    chunk_size_words = max(1, int(chunk_size_tokens / token_per_word))
    overlap_word_count = (
        max(0, overlap_words)
        if overlap_words is not None
        else max(0, int(overlap_tokens / token_per_word))
    )

    if max_input_tokens and max_input_tokens > 0:
        estimated_total_tokens = math.ceil(len(words) * token_per_word)
        if estimated_total_tokens <= max_input_tokens:
            return [text]
        chunk_size_words = max(1, int(max_input_tokens / token_per_word))
        # Ensure we can advance while retaining approximately 1000 tokens of overlap.
        if chunk_size_words <= overlap_word_count:
            overlap_word_count = max(0, chunk_size_words - 1)

    if len(words) <= chunk_size_words:
        return [text]
    
    chunks = []
    start = 0
    step = max(1, chunk_size_words - overlap_word_count)
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step
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


LOCAL_FULL_TEXT_DIRS = [
    Path("."),
    Path("FullTexts"),
]
LOCAL_FULL_TEXT_EXTENSIONS = [".txt", ".TXT", ".pdf", ".PDF"]


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pypdf/PyPDF2 when available."""
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF import requires pypdf or PyPDF2.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_local_full_text(pmid: str, search_dirs: Optional[List[Path]] = None) -> Optional[str]:
    """Load a locally available PMID-indexed TXT/PDF full-text file if present."""
    search_dirs = search_dirs or LOCAL_FULL_TEXT_DIRS
    candidate_names = []
    for ext in LOCAL_FULL_TEXT_EXTENSIONS:
        candidate_names.extend([f"{pmid}{ext}", f"PMID{pmid}{ext}", f"pmid_{pmid}{ext}"])

    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in candidate_names:
            path = directory / name
            if path.exists() and path.is_file():
                if path.suffix.lower() == ".pdf":
                    try:
                        text = extract_pdf_text(path)
                    except Exception as exc:
                        console.print(f"[red]Failed to extract PDF text from {path}: {exc}[/]")
                        continue
                else:
                    try:
                        text = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                clean = clean_text_v2(text)
                if clean:
                    console.print(f"[green]Loaded local full text for PMID {pmid}: {path}[/]")
                    return clean
    return None

# =====================================
# 5. Prompt Definitions
# =====================================
def get_prompts():
    """Define the four-stage prompt workflow for schema-aligned JSON extraction."""
    prompt1 = """
You are a medical research assistant specializing in clinical risk prediction model literature.

General extraction rules:
1. Extract information only from the provided article text. Do not use outside knowledge, memory, or assumptions.
2. If a field is not explicitly reported, answer N/A. If the field cannot be determined with confidence, answer N/A rather than guessing.
3. Only perform direct arithmetic calculations when all required source values are explicitly reported in the text. Do not infer, estimate, impute, or back-calculate missing values.
4. Distinguish development/internal validation from external validation. External validation requires an independent cohort, institution, registry, geographic population, or clearly independent time period. Random train-test splits, cross-validation, bootstrap validation, and internal test sets are not external validation.
5. Study type should be extracted as reported by the authors and should not be forced into a prospective/retrospective binary.

[Extracted text]

Stage 1: Comprehension and evidence extraction for study and sample information.
Read the article and extract evidence for the following fields:
Study Type; Disease Name; External Validation; Date Range in the Development Set; Date Range in the Validation Set; Median Follow-up Time (Years) and IQR; Mean Follow-up Time (Years) and Standard Error; Data Sources; Sample Characteristics; Number of Cases in the Development Set; Number of Controls in the Development Set; Number of Cases in the Validation Set; Number of Controls in the Validation Set; Number of Female Participants (Development); Number of Female Participants (Validation); Number of Male Participants (Development); Number of Male Participants (Validation); Age Range (Development); Age Range (Validation); Average Age and Standard Deviation (Development); Average Age and Standard Deviation (Validation); Median Age (Development) and IQR; Median Age (Validation) and IQR; Racial/Ethnic Composition (Development); Racial/Ethnic Composition (Validation).

Return concise field-level notes with supporting evidence phrases or source sections where possible. Do not output JSON in this stage.
    """
    prompt2 = """
You are continuing the same clinical prediction model extraction task.

[Extracted text]

Stage 2: Comprehension and evidence extraction for prediction-model information.
Identify every distinct prediction model and every distinct model stage reported in the article. A model-stage record should correspond to either Development/internal validation or External validation.

For each model-stage record, extract evidence for:
Model Numbers; Model Stage; Model Name; Model Type; Prediction Variables; AUC Values; C-index Values; Accuracy; F1-score; Calibration Values; Nomogram Application; Use of TRIPOD Guidelines.

Rules:
1. Do not merge development/internal validation metrics with external validation metrics.
2. Do not prioritize validation metrics over development metrics; instead create separate model-stage records when both are reported.
3. If the article reports multiple models, include all primary, secondary, comparative, and machine-learning models that have extractable model information.
4. Model Stage must be exactly one of: Development/internal validation; External validation.
5. Model Type must describe the modeling method only, such as Logistic regression, Cox proportional hazards model, Random forest, Support vector machine, Neural network, Nomogram, Polygenic risk score. Do not write validation status as model type.

Return concise model-level notes with supporting evidence phrases or source sections where possible. Do not output JSON in this stage.
    """
    prompt3 = """
Stage 3: Convert the study and sample information into schema-constrained JSON.

Use only the article text and the prior extraction notes below. Do not add new facts.

[Prior responses]

Return exactly one JSON object and no extra text. Use this exact schema and these exact keys:
{
  "Study Type": "string",
  "Disease Name": "string",
  "External Validation": "Yes or No",
  "Date Range in the Development Set": "string",
  "Date Range in the Validation Set": "string",
  "Median Follow-up Time (Years) and IQR": "string",
  "Mean Follow-up Time (Years) and Standard Error": "string",
  "Data Sources": "string",
  "Sample Characteristics": "string",
  "Number of Cases in the Development Set": "string",
  "Number of Controls in the Development Set": "string",
  "Number of Cases in the Validation Set": "string",
  "Number of Controls in the Validation Set": "string",
  "Number of Female Participants (Development)": "string",
  "Number of Female Participants (Validation)": "string",
  "Number of Male Participants (Development)": "string",
  "Number of Male Participants (Validation)": "string",
  "Age Range (Development)": "string",
  "Age Range (Validation)": "string",
  "Average Age and Standard Deviation (Development)": "string",
  "Average Age and Standard Deviation (Validation)": "string",
  "Median Age (Development) and IQR": "string",
  "Median Age (Validation) and IQR": "string",
  "Racial/Ethnic Composition (Development)": "string",
  "Racial/Ethnic Composition (Validation)": "string"
}

Use "N/A" for unreported or uncertain fields.
    """
    prompt4 = """
Stage 4: Convert the prediction-model information into a schema-constrained JSON array.

Use only the article text and the prior extraction notes below. Do not add new facts.

[Prior responses]

Return exactly one JSON array and no extra text. Each array element must use this exact schema and these exact keys:
{
  "Model Numbers": "string",
  "Model Stage": "Development/internal validation or External validation",
  "Model Name": "string",
  "Model Type": "string",
  "Prediction Variables": "string",
  "AUC Values": "string",
  "C-index Values": "string",
  "Accuracy": "string",
  "F1-score": "string",
  "Calibration Values": "string",
  "Nomogram Application": "Yes or No",
  "Use of TRIPOD Guidelines": "Yes or No"
}

Rules:
1. Model Stage must be exactly "Development/internal validation" or "External validation".
2. Create separate records when the same model has separate development/internal-validation and external-validation results.
3. Do not use validation status as Model Type.
4. Use "N/A" for unreported or uncertain fields.
    """
    return [prompt1, prompt2, prompt3, prompt4]

# =====================================
# 6. Structured Data Storage
# =====================================
import csv
import json

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
    "Model Numbers",
    "Model Stage",
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

MODEL_STAGE_ALLOWED_VALUES = {"Development/internal validation", "External validation"}

def normalize_missing_value(value):
    """Normalize missing values to N/A while preserving reported strings."""
    if value is None:
        return "N/A"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    value = str(value).strip()
    if not value or value.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return "N/A"
    return value

def extract_json_payload(response_text):
    """
    Parse a JSON object/array from an LLM response.

    The primary path is strict json.loads(). A limited rescue path extracts the
    first fenced JSON block or the outermost JSON object/array when the model
    accidentally wraps the payload in prose.
    """
    if not response_text:
        raise ValueError("Empty LLM response")

    text = response_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1).strip())

    starts = [idx for idx in [text.find("{"), text.find("[")] if idx != -1]
    if not starts:
        raise ValueError("No JSON object or array found in response")
    start = min(starts)
    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        raise ValueError("Malformed JSON payload")
    return json.loads(text[start:end + 1])

def validate_study_sample_json(payload, pmid):
    """Validate and normalize the Study & Sample Information JSON object."""
    if not isinstance(payload, dict):
        raise ValueError("Study/sample JSON must be an object")

    record = {field: "N/A" for field in sample_fields}
    record["PMID"] = pmid
    for field in sample_fields:
        if field == "PMID":
            continue
        record[field] = normalize_missing_value(payload.get(field, "N/A"))

    if record["External Validation"] not in {"Yes", "No", "N/A"}:
        value = record["External Validation"].lower()
        if value in {"yes", "y", "true"}:
            record["External Validation"] = "Yes"
        elif value in {"no", "n", "false"}:
            record["External Validation"] = "No"
        else:
            record["External Validation"] = "N/A"
    return record

def validate_model_json(payload, pmid):
    """Validate and normalize the Model Information JSON array."""
    if isinstance(payload, dict):
        if "model_information" in payload:
            payload = payload["model_information"]
        elif "models" in payload:
            payload = payload["models"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Model JSON must be an array")

    records = []
    for idx, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            continue
        record = {field: "N/A" for field in model_fields}
        record["PMID"] = pmid
        for field in model_fields:
            if field == "PMID":
                continue
            record[field] = normalize_missing_value(item.get(field, "N/A"))

        if record["Model Numbers"] == "N/A":
            record["Model Numbers"] = str(idx)
        if record["Model Stage"] not in MODEL_STAGE_ALLOWED_VALUES:
            stage = record["Model Stage"].lower()
            if "external" in stage:
                record["Model Stage"] = "External validation"
            else:
                record["Model Stage"] = "Development/internal validation"
        records.append(record)

    if not records:
        empty_record = {field: "N/A" for field in model_fields}
        empty_record["PMID"] = pmid
        empty_record["Model Numbers"] = "1"
        empty_record["Model Stage"] = "Development/internal validation"
        records.append(empty_record)
    return records

def process_input(input_text, prompts, max_input_tokens: Optional[int] = None):
    """Process input text and get LLM responses using prompts"""
    responses = []
    text_chunks = chunk_text_v2(
        input_text,
        max_input_tokens=max_input_tokens,
        overlap_tokens=1000
    )
    if not text_chunks:
        text_chunks = [input_text]

    if len(text_chunks) > 1:
        console.print(f"[yellow]Input text exceeds token limit estimate, split into {len(text_chunks)} chunks (approximately 1000-token overlap).[/]")

    for i, prompt in enumerate(prompts, 1):
        console.print(f"[italic yellow]Processing Prompt {i}...[/]")
        prior_context = "\n\n".join(
            f"Stage {stage_idx} response:\n{response}"
            for stage_idx, response in enumerate(responses, 1)
        )

        if "[Extracted text]" in prompt and len(text_chunks) > 1:
            chunk_responses = []
            for chunk_idx, chunk_text in enumerate(text_chunks, 1):
                console.print(f"[dim]  Prompt {i}: chunk {chunk_idx}/{len(text_chunks)}[/]")
                prompt_text = prompt.replace("[Extracted text]", chunk_text)
                prompt_text = prompt_text.replace("[Prior responses]", prior_context)
                messages = [{"role": "user", "content": prompt_text}]
                chunk_response = chat_completion(messages)
                if chunk_response:
                    chunk_responses.append(f"[Chunk {chunk_idx}/{len(text_chunks)}]\n{chunk_response}")

            if chunk_responses:
                combined_response = "\n\n".join(chunk_responses)
                responses.append(combined_response)
                console.print(f"\n[bold magenta]Response {i}:[/]")
                console.print(combined_response)
                console.print("\n" + "-"*50 + "\n")
            else:
                console.print(f"[bold red]Prompt {i} processing failed[/]")
        else:
            prompt_text = prompt.replace("[Extracted text]", input_text)
            prompt_text = prompt_text.replace("[Prior responses]", prior_context)
            messages = [{"role": "user", "content": prompt_text}]
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
    """Extract study/sample data from schema-constrained JSON."""
    payload = extract_json_payload(response_sample)
    return validate_study_sample_json(payload, pmid)

def extract_model_data(response_model, pmid):
    """Extract model-stage data from schema-constrained JSON."""
    payload = extract_json_payload(response_model)
    return validate_model_json(payload, pmid)

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
    full_text_source = choose_full_text_source()

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
                
                if full_text_source == "pmc" and pubmed_data.get('pmcid'):
                    full_text_data = fetch_pmc_full_text_v2(pubmed_data['pmcid'], driver)
                    full_text = full_text_data.get('full_text', "")
                elif full_text_source == "pmc":
                    console.print(f"[yellow]No PMCID found for PMID {pmid}; online PMC retrieval unavailable.[/]")
                    full_text = ""
                else:
                    full_text = load_local_full_text(pmid) or ""
                
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
                    responses = process_input(full_text, prompts, max_input_tokens=API_MAX_INPUT_TOKENS)
                    
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
