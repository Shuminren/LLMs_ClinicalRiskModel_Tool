import csv
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# 默认 PMID 列表文件（注意：在 1st Revision 根目录）
DEFAULT_PMID_CSV = "../pmid.csv"


@dataclass
class PubMedRecord:
    PMID: str
    Journal_Information: Optional[str] = None
    Publication_Year: Optional[str] = None
    Publication_Country: Optional[str] = None
    DOI_Number: Optional[str] = None
    Title: Optional[str] = None
    Author_List: Optional[str] = None
    Key_Words: Optional[str] = None


def fetch_html(pmid: str) -> Optional[str]:
    url = BASE_URL.format(pmid=pmid)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        # 确保正确设置编码，处理特殊字符（如 ø）
        resp.encoding = resp.apparent_encoding or 'utf-8'
    except Exception as e:
        print(f"PMID {pmid}: 请求出错 {e}", file=sys.stderr)
        return None

    if resp.status_code != 200:
        print(f"PMID {pmid}: HTTP {resp.status_code}", file=sys.stderr)
        return None
    return resp.text


def parse_journal_and_year(soup: BeautifulSoup) -> (Optional[str], Optional[str]):
    """
    解析期刊名和出版年份。
    优先从 meta 标签 citation_journal_title 中提取期刊名（最可靠）。
    参考页面结构见示例文献：Heliyon 文章（PMID 37842633）[https://pubmed.ncbi.nlm.nih.gov/37842633/]
    """
    journal = None
    year = None

    # 方法 1：优先使用 meta 标签 citation_journal_title（最可靠）
    meta_journal = soup.find("meta", attrs={"name": "citation_journal_title"})
    if meta_journal and meta_journal.get("content"):
        journal = meta_journal["content"].strip()

    # 方法 2：如果没有 meta 标签，尝试从 citation 文本中提取
    if not journal:
        citation = None
        cit_el = soup.select_one("div.citation .cit, div.citation, .cit")
        if cit_el:
            citation = cit_el.get_text(" ", strip=True)

        if citation:
            # 期刊名：在第一个句号之前，但要排除日期开头的情况
            # 例如："Heliyon. 2023 Oct 5;9(10):e20684."
            # 如果 citation 以年份开头（如 "2023 Oct..."），则跳过此方法
            if not re.match(r"^\s*(19|20)\d{2}", citation):
                parts = citation.split(".")
                if parts and parts[0].strip():
                    # 验证第一部分不是纯数字或日期格式
                    first_part = parts[0].strip()
                    if not re.match(r"^\s*(19|20)\d{2}", first_part) and not re.match(r"^\d+$", first_part):
                        journal = first_part

    # 方法 3：尝试从页面标题区域查找
    if not journal:
        # 查找包含期刊名的其他位置
        journal_els = soup.select("span.journal-name, .journal-title, cite.journal")
        for el in journal_els:
            text = el.get_text(" ", strip=True)
            if text and len(text) < 100:  # 期刊名通常不会太长
                journal = text
                break

    # 解析年份
    meta_year = soup.find("meta", attrs={"name": "citation_date"})
    if meta_year and meta_year.get("content"):
        m = re.search(r"(19|20)\d{2}", meta_year["content"])
        if m:
            year = m.group(0)
    
    # 如果 meta 中没有年份，从 citation 或其他位置查找
    if not year:
        citation = None
        cit_el = soup.select_one("div.citation .cit, div.citation, .cit")
        if cit_el:
            citation = cit_el.get_text(" ", strip=True)
        if citation:
            m = re.search(r"(19|20)\d{2}", citation)
            if m:
                year = m.group(0)

    return journal, year


def parse_country(soup: BeautifulSoup) -> Optional[str]:
    """
    从第一单位（affiliation）的最后一个词中提取国家。
    例如：示例文献中第一单位为
    “School of Computer Science, Nanjing Audit University, China.”
    -> Publication_Country = "China"
    """
    # PubMed 新界面 affiliations 结构
    aff_el = soup.select_one("div.affiliations li, ul.affiliations li, div.affiliation")
    if not aff_el:
        return None
    text = aff_el.get_text(" ", strip=True)
    if not text:
        return None
    # 去掉句号等标点，取最后一个词
    text = re.sub(r"[.;]+$", "", text)
    parts = [p for p in re.split(r"[,\s]+", text) if p]
    if not parts:
        return None
    return parts[-1]


def parse_doi(soup: BeautifulSoup) -> Optional[str]:
    # 1) 标准 DOI 区域
    doi_el = soup.find("span", class_="identifier", attrs={"data-ga-label": "DOI"})
    if doi_el:
        t = doi_el.get_text(" ", strip=True)
        m = re.search(r"10\.\S+\/\S+", t)
        if m:
            return m.group(0)

    # 2) 任何包含 “doi:” 文本的地方
    possible = soup.find_all(string=re.compile(r"doi:", re.I))
    for s in possible:
        m = re.search(r"doi:\s*(10\.\S+\/\S+)", s, re.I)
        if m:
            return m.group(1)

    # 3) meta 标签
    meta = soup.find("meta", attrs={"name": "citation_doi"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return None


def parse_title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    meta = soup.find("meta", attrs={"name": "citation_title"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return None


def clean_author_name(name: str) -> str:
    # 去掉末尾的数字、逗号
    name = re.sub(r"\s*\d+$", "", name)
    return name.strip(" ,")


def parse_authors(soup: BeautifulSoup) -> Optional[str]:
    """
    从作者列表中提取姓名，并移除姓名后面的数字。
    比如示例文献中：
    "Jiatong Han  1, Hao Li  1, Han Lin  2, ..." -> "Jiatong Han, Hao Li, Han Lin, ..."
    """
    # 1) 页面上作者列表（使用 set 去重，因为页面可能有多个作者列表区域）
    names_set = set()
    names_list = []

    # 查找所有 authors-list 容器
    authors_containers = soup.select("div.authors-list, .authors-list")

    # 只使用第一个容器，避免重复
    if authors_containers:
        container = authors_containers[0]
        for sel in ["a.full-name", "span.author-name", ".full-name"]:
            els = container.select(sel)
            if els:
                for el in els:
                    n = el.get_text(" ", strip=True)
                    if n:
                        cleaned = clean_author_name(n)
                        if cleaned and cleaned not in names_set:
                            names_set.add(cleaned)
                            names_list.append(cleaned)
                if names_list:
                    break

    # 2) 退而求其次：从 "Authors" 文本块正则拆
    if not names_list:
        auth_block = soup.find(string=re.compile(r"Authors", re.I))
        if auth_block:
            txt = auth_block.parent.get_text(" ", strip=True)
            m = re.search(r"Authors?:\s*(.+)", txt, re.I)
            if m:
                raw = m.group(1)
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                for p in parts:
                    cleaned = clean_author_name(p)
                    if cleaned and cleaned not in names_set:
                        names_set.add(cleaned)
                        names_list.append(cleaned)

    if not names_list:
        return None
    return ", ".join(names_list)


def parse_keywords(soup: BeautifulSoup) -> Optional[str]:
    """
    提取 Key_Words，例如示例文献中的：
    "CHARLS; Characteristic variables; Depression; LassoNet-RNN"
    """

    # 方法 1: 直接在 Abstract 部分查找 Keywords（最常见的结构）
    abstract_section = soup.find("div", class_="abstract-content") or soup.find("section", class_="abstract") or soup.find("div", class_="abstract")
    if abstract_section:
        abstract_text = abstract_section.get_text(" ", strip=True)
        # 从文本末尾查找 Keywords: 模式
        m = re.search(r"Keywords?:\s*(.+)$", abstract_text, re.I | re.MULTILINE)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r"[.;]+$", "", raw)
            if raw and len(raw) > 3:
                if ";" in raw:
                    return raw
                kws = [kw.strip() for kw in re.split(r"[,;]\s*", raw) if kw.strip()]
                if kws:
                    return "; ".join(kws)

    # 方法 2: 从 meta 标签中提取
    meta_kw = soup.find("meta", attrs={"name": "citation_keywords"})
    if meta_kw and meta_kw.get("content"):
        content = meta_kw["content"].strip()
        if content:
            keywords_list = [kw.strip() for kw in re.split(r"[;,\n]+", content) if kw.strip()]
            if keywords_list:
                return "; ".join(keywords_list)

    # 方法 3: 查找包含 "Keywords" 的 strong 标签
    strong_tags = soup.find_all("strong")
    for strong_kw in strong_tags:
        strong_text = strong_kw.get_text(strip=True)
        if re.search(r"Keywords?", strong_text, re.I):
            parent = strong_kw.parent
            if parent:
                text = parent.get_text(" ", strip=True)
                m = re.search(r"Keywords?:\s*(.+)", text, re.I)
                if m:
                    raw = m.group(1).strip()
                    raw = re.sub(r"[.;]+$", "", raw)
                    if raw and len(raw) > 3:
                        if ";" in raw:
                            return raw
                        kws = [kw.strip() for kw in re.split(r"[,;]\s*", raw) if kw.strip()]
                        if kws:
                            return "; ".join(kws)

    return None


def parse_record(pmid: str, html: str) -> PubMedRecord:
    soup = BeautifulSoup(html, "html.parser")

    journal, year = parse_journal_and_year(soup)
    country = parse_country(soup)
    doi = parse_doi(soup)
    title = parse_title(soup)
    authors = parse_authors(soup)
    keywords = parse_keywords(soup)

    return PubMedRecord(
        PMID=pmid,
        Journal_Information=journal,
        Publication_Year=year,
        Publication_Country=country,
        DOI_Number=doi,
        Title=title,
        Author_List=authors,
        Key_Words=keywords,
    )


def save_to_csv(records: List[PubMedRecord], filename: str) -> None:
    fieldnames = [
        "PMID",
        "Journal_Information",
        "Publication_Year",
        "Publication_Country",
        "DOI_Number",
        "Title",
        "Author_List",
        "Key_Words",
    ]
    # 使用 utf-8-sig 编码，确保 Excel 等工具可以正确打开包含特殊字符的 CSV
    # 这样可以正确处理像 "Tjønneland" 中的 ø 这样的特殊字符
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            # 确保所有字符串值都是 Unicode 字符串，正确处理特殊字符
            row_dict = {}
            for key, value in asdict(r).items():
                if value is not None:
                    # 确保值是字符串且正确处理编码
                    row_dict[key] = str(value)
                else:
                    row_dict[key] = ""
            writer.writerow(row_dict)


def read_pmids_from_csv(csv_path: str) -> List[str]:
    """
    从 pmid.csv 中读取 PMID 列表。
    兼容两种常见格式：
    1) 带表头：第一列或某列名为 PMID
    2) 无表头：每行第一列为 PMID
    """
    pmids: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        pmid_col_idx = 0

        if header:
            # 判断第一行是不是表头
            lowered = [h.strip().lower() for h in header]
            if "pmid" in lowered:
                pmid_col_idx = lowered.index("pmid")
            else:
                # 第一行当作数据行处理
                row = header
                if row and row[0].strip():
                    pmids.append(row[0].strip())

            for row in reader:
                if not row:
                    continue
                if len(row) <= pmid_col_idx:
                    continue
                v = row[pmid_col_idx].strip()
                if v:
                    pmids.append(v)
        else:
            # 空文件
            pass

    return pmids


def main(argv: List[str]) -> None:
    """
    推荐用法（在 codes 目录下）：
      python pubmed_scraper.py
    默认：
      - 输入 PMID 文件：../pmid.csv
      - 输出结果文件：pubmed_result.csv

    其他用法：
    1) 指定输出文件：
       python pubmed_scraper.py pubmed_output.csv
       -> 读取 ../pmid.csv，输出到 pubmed_output.csv

    2) 同时指定输入和输出：
       python pubmed_scraper.py ../pmid.csv pubmed_output.csv
    """
    if len(argv) == 0:
        in_file = DEFAULT_PMID_CSV
        out_file = "pubmed_result.csv"
    elif len(argv) == 1:
        in_file = DEFAULT_PMID_CSV
        out_file = argv[0]
    else:
        in_file = argv[0]
        out_file = argv[1]

    try:
        pmids = read_pmids_from_csv(in_file)
    except FileNotFoundError:
        print(f"未找到 PMID 文件: {in_file}", file=sys.stderr)
        sys.exit(1)

    if not pmids:
        print(f"{in_file} 中未读取到任何 PMID。", file=sys.stderr)
        sys.exit(1)

    records: List[PubMedRecord] = []
    for pmid in pmids:
        html = fetch_html(pmid)
        if not html:
            continue
        record = parse_record(pmid, html)
        records.append(record)
        print(f"已解析 PMID {pmid}", file=sys.stderr)

    if not records:
        print("未成功解析任何 PMID。", file=sys.stderr)
        sys.exit(1)

    save_to_csv(records, out_file)
    print(f"已保存 {len(records)} 条记录到 {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

import csv
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


@dataclass
class PubMedRecord:
    PMID: str
    Journal_Information: Optional[str] = None
    Publication_Year: Optional[str] = None
    Publication_Country: Optional[str] = None
    DOI_Number: Optional[str] = None
    Title: Optional[str] = None
    Author_List: Optional[str] = None
    Key_Words: Optional[str] = None


def fetch_html(pmid: str) -> Optional[str]:
    url = BASE_URL.format(pmid=pmid)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        print(f"PMID {pmid}: HTTP {resp.status_code}", file=sys.stderr)
        return None
    return resp.text


def parse_country(soup: BeautifulSoup) -> Optional[str]:
    """
    从第一单位（affiliation）的最后一个词中提取国家。
    例如：“School of Computer Science, Nanjing Audit University, China.”
    """
    # PubMed 新界面 affiliations 结构
    aff_el = soup.select_one("div.affiliations li, ul.affiliations li, div.affiliation")
    if not aff_el:
        return None
    text = aff_el.get_text(" ", strip=True)
    if not text:
        return None
    # 去掉句号等标点，取最后一个词
    text = re.sub(r"[.;]+$", "", text)
    parts = [p for p in re.split(r"[,\s]+", text) if p]
    if not parts:
        return None
    return parts[-1]


def parse_doi(soup: BeautifulSoup) -> Optional[str]:
    # 1) 标准 DOI 区域
    doi_el = soup.find("span", class_="identifier", attrs={"data-ga-label": "DOI"})
    if doi_el:
        t = doi_el.get_text(" ", strip=True)
        m = re.search(r"10\.\S+\/\S+", t)
        if m:
            return m.group(0)

    # 2) 任何包含 “doi:” 文本的地方
    possible = soup.find_all(string=re.compile(r"doi:", re.I))
    for s in possible:
        m = re.search(r"doi:\s*(10\.\S+\/\S+)", s, re.I)
        if m:
            return m.group(1)

    # 3) meta 标签
    meta = soup.find("meta", attrs={"name": "citation_doi"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return None


def parse_title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    meta = soup.find("meta", attrs={"name": "citation_title"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return None


def clean_author_name(name: str) -> str:
    # 去掉末尾的数字、逗号
    name = re.sub(r"\s*\d+$", "", name)
    return name.strip(" ,")


def parse_authors(soup: BeautifulSoup) -> Optional[str]:
    # 1) 页面上作者列表
    names: List[str] = []

    for sel in [
        "div.authors-list span.author-name",
        "div.authors-list a.full-name",
        ".authors-list .full-name",
    ]:
        els = soup.select(sel)
        if els:
            for el in els:
                n = el.get_text(" ", strip=True)
                if n:
                    names.append(clean_author_name(n))
            break

    # 2) 退而求其次：从 “Authors” 文本块正则拆
    if not names:
        auth_block = soup.find(string=re.compile(r"Authors", re.I))
        if auth_block:
            txt = auth_block.parent.get_text(" ", strip=True)
            m = re.search(r"Authors?:\s*(.+)", txt, re.I)
            if m:
                raw = m.group(1)
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                names = [clean_author_name(p) for p in parts]

    if not names:
        return None
    return ", ".join(names)


def read_pmids_from_csv(csv_path: str) -> List[str]:
    """
    从 pmid.csv 中读取 PMID 列表。
    兼容两种常见格式：
    1) 带表头：第一列或某列名为 PMID
    2) 无表头：每行第一列为 PMID
    """
    pmids: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        pmid_col_idx = 0

        if header:
            # 判断第一行是不是表头
            lowered = [h.strip().lower() for h in header]
            if "pmid" in lowered:
                pmid_col_idx = lowered.index("pmid")
            else:
                # 第一行当作数据行处理
                row = header
                if row and row[0].strip():
                    pmids.append(row[0].strip())

            for row in reader:
                if not row:
                    continue
                if len(row) <= pmid_col_idx:
                    continue
                v = row[pmid_col_idx].strip()
                if v:
                    pmids.append(v)
        else:
            # 空文件
            pass

    return pmids


def main(argv: List[str]) -> None:
    """
    使用方式：
    1) 默认：当前目录存在 pmid.csv
       python pubmed_scraper.py
       -> 读取 pmid.csv，输出 pubmed_result.csv

    2) 指定输出文件：
       python pubmed_scraper.py output.csv
       -> 读取 pmid.csv，输出到 output.csv

    3) 同时指定输入和输出：
       python pubmed_scraper.py pmid.csv output.csv
    """
    if len(argv) == 0:
        in_file = "pmid.csv"
        out_file = "pubmed_result.csv"
    elif len(argv) == 1:
        in_file = "pmid.csv"
        out_file = argv[0]
    else:
        in_file = argv[0]
        out_file = argv[1]

    try:
        pmids = read_pmids_from_csv(in_file)
    except FileNotFoundError:
        print(f"未找到 PMID 文件: {in_file}", file=sys.stderr)
        sys.exit(1)

    if not pmids:
        print(f"{in_file} 中未读取到任何 PMID。", file=sys.stderr)
        sys.exit(1)

    records: List[PubMedRecord] = []
    for pmid in pmids:
        html = fetch_html(pmid)
        if not html:
            continue
        record = parse_record(pmid, html)
        records.append(record)
        print(f"已解析 PMID {pmid}", file=sys.stderr)

    if not records:
        print("未成功解析任何 PMID。", file=sys.stderr)
        sys.exit(1)

    save_to_csv(records, out_file)
    print(f"已保存 {len(records)} 条记录到 {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])


