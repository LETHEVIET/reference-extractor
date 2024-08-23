from pypdf import PdfReader
import subprocess
import os
import uuid
import re
import json
import pathlib
from typing import List
from pdfminer.high_level import extract_pages
import re
from pdfminer.layout import LTTextBoxHorizontal, LTTextBox, LTTextLine, LTChar
import requests
import requests
from arxvi_utils import parse_arxiv_response
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

ANYSTYPE_PATH="/home/viet/.local/share/gem/ruby/3.2.0/bin/anystyle"
ARXIV_PDF_URL = "http://export.arxiv.org/pdf/"
CROSSEF_SEARCH_URL = "https://api.crossref.org/works"

def remove_hyphens(text: str) -> str:
    """

    This fails for:
    * Natural dashes: well-known, self-replication, use-cases, non-semantic,
                      Post-processing, Window-wise, viewpoint-dependent
    * Trailing math operands: 2 - 4
    * Names: Lopez-Ferreras, VGG-19, CIFAR-100
    """
    lines = [line.rstrip() for line in text.split("\n")]

    # Find dashes
    line_numbers = []
    for line_no, line in enumerate(lines[:-1]):
        if line.endswith("-"):
            line_numbers.append(line_no)

    # Replace
    for line_no in line_numbers:
        lines = dehyphenate(lines, line_no)

    return "\n".join(lines)

def dehyphenate(lines: List[str], line_no: int) -> List[str]:
    next_line = lines[line_no + 1]
    word_suffix = next_line.split(" ")[0]

    lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = lines[line_no + 1][len(word_suffix) :]
    return lines

def replace_ligatures(text: str) -> str:
    ligatures = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        # "Ꜳ": "AA",
        # "Æ": "AE",
        "ꜳ": "aa",
    }
    for search, replace in ligatures.items():
        text = text.replace(search, replace)
    return text

def clean_text(text):
    text = replace_ligatures(text)
    text = remove_hyphens(text)
    return text

def extract_references_and_citations(text):
    
    # Find the reference section
    text = clean_text(text)
    reference_section = re.search(r'References([\s\S]*)', text, re.IGNORECASE)
    if reference_section:
        references = reference_section.group(1)
    else:
        return "References section not found."
    
    # Extract individual references
    reference_list = re.findall(r'\[\d+\].*?(?=\[\d+\]|\Z)', references, re.DOTALL)

    reference_list = [ref.replace('\n','') for ref in reference_list]
    
    return reference_list

def parse_pdf2text(pdf_path):
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

def parse_references(references, work_dir):
    file_path = f"{work_dir}/refs.txt"
    with open(file_path, "w") as f:
        for ref in references:
            f.write(ref[4:] + '\n')

    process = subprocess.Popen([ANYSTYPE_PATH, 'parse', file_path], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           text=True)

    stdout, stderr = process.communicate()

    if stderr:
        raise NameError(stderr)

    parsed_refs = json.loads(stdout)

    for i, ref in enumerate(references):
        parsed_refs[i]["cite_id"] = ref[1:2]

    return parsed_refs

def make_working_dir():
    paper_uuid = uuid.uuid4()
    working_dir = f"tmp/{paper_uuid}/"
    pathlib.Path(working_dir).mkdir(parents=True, exist_ok=True)
    return working_dir

def tfidf_similarity(text1, text2):
    # Create TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0]

def crossref_search(reference):
    params = {
        'query': reference
    }
    r = requests.get(CROSSEF_SEARCH_URL, params= params)
    
    return r.json()['message']['items'][0]

def get_paper_link(doi):
    URL = f"https://doi.org/{doi}" # Specify the DOI here
    r = requests.get(URL,allow_redirects=True) # Redirects help follow to the actual domain
    return r.url

def arxiv_search(reference):
    params = {
        "search_query": reference,
        "start": 0,
        "max_results": 1,
    }

    response = requests.get("http://export.arxiv.org/api/query", params=params) 
    parsed_res = parse_arxiv_response(response.text)
    if len(parsed_res) != 0:
        return parsed_res[0]
    else:
        return {
            "title": "",
            "id": ""
        }

def retrieve_from_crossref(parsed_ref):
    search_results = {
        'tf-idf_score': [],
        'ref_title': [],
        'res_title': [],
        'res_DOI': [],
        'paper_link': []
    }

    for ref in tqdm(parsed_ref):
        ref_text_title = ref["title"][0]
        search_result = crossref_search(ref_text_title)
        search_result_title = search_result['title'][0]
        doi = search_result['DOI']
        paper_link = get_paper_link(doi)
        sim_score = tfidf_similarity(ref_text_title, search_result_title)

        search_results['ref_title'].append(ref_text_title)
        search_results['res_title'].append(search_result_title)
        search_results['res_DOI'].append(doi)
        search_results['tf-idf_score'].append(sim_score)
        search_results['paper_link'].append(paper_link)
        
        # print(f"{sim_score} | {ref_text_title} | {search_result_title}")


    retrieve_df = pd.DataFrame(search_results)
    # retrieve_df.to_csv("tmp/retrieve_table.csv")
    return retrieve_df

def retrieve_from_arxiv(parsed_ref):
    arvix_search_results = {
        'cite_id': [],
        'tf-idf_score': [],
        'ref_title': [],
        'res_title': [],
        'arxiv_id': []
    }

    for ref in tqdm(parsed_ref):
        ref_text_title = ref["title"][0]
        search_result = arxiv_search(ref_text_title)
        search_result_title = search_result['title']
        search_result_id = search_result['id'].split("/")[-1]
        sim_score = tfidf_similarity(ref_text_title, search_result_title)
        
        arvix_search_results['cite_id'].append(ref['cite_id'])
        arvix_search_results['ref_title'].append(ref_text_title)
        arvix_search_results['res_title'].append(search_result_title)
        arvix_search_results['tf-idf_score'].append(sim_score)
        arvix_search_results['arxiv_id'].append(search_result_id)

    retrieve_df = pd.DataFrame(arvix_search_results)
    # retrieve_df.to_csv("tmp/arvix_retrieve_table.csv")

    return retrieve_df

def retrieve_from_crossref(parsed_ref):
    arvix_search_results = {
        'cite_id': [],
        'tf-idf_score': [],
        'ref_title': [],
        'res_title': [],
        'URL': []
    }

    for ref in tqdm(parsed_ref):
        ref_text_title = ref["title"][0]
        search_result = crossref_search(ref_text_title)
        search_result_title = search_result['title'][0]
        search_result_url = search_result['URL']
        sim_score = tfidf_similarity(ref_text_title, search_result_title)
        
        arvix_search_results['cite_id'].append(ref['cite_id'])
        arvix_search_results['ref_title'].append(ref_text_title)
        arvix_search_results['res_title'].append(search_result_title)
        arvix_search_results['tf-idf_score'].append(sim_score)
        arvix_search_results['URL'].append(search_result_url)

    retrieve_df = pd.DataFrame(arvix_search_results)

    return retrieve_df

def download_papers(retrieve_df, working_dir):
    download_path = f"{working_dir}/papers"
    mapper = {}
    pathlib.Path(download_path).mkdir(exist_ok=True)
    for index, row in tqdm(retrieve_df.iterrows()):
        arxiv_id = row["arxiv_id"]
        cite_id = row["cite_id"]
        pdf_url = f"{ARXIV_PDF_URL}{arxiv_id}"
        response = requests.get(pdf_url)

        save_path = f"{download_path}/{arxiv_id}.pdf"
        with open(save_path, "wb") as f:
            f.write(response.content) 

        mapper[cite_id] = save_path

    return mapper

def is_reference_header(text):
    """Check if the given text is likely to be a reference section header."""
    reference_headers = ['references', 'bibliography', 'works cited', 'literature cited']
    return any(header in text.lower() for header in reference_headers)

def extract_citation(text):
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'

    citations = re.findall(citation_pattern, text)
    citation_sentences = []
    if citations:
        # Split text into sentences (simple split on period)
        sentences = text.split('.')
        for sentence in sentences:
            if re.search(citation_pattern, sentence):
                # cleaned_text = clean_text(sentence.strip())
                # cleaned_text = cleaned_text.replace('\n', ' ').replace('- ', '')
                cleaned_text = sentence
                citation_sentences.append(cleaned_text)

        r_citations = []
        r_citation_sentences = []

        for sentence in citation_sentences:
            cites = re.findall(citation_pattern, sentence)
            for citation in cites:
                splitted_cites = citation.split(', ')
                for cite in splitted_cites:
                    r_citations.append(cite)
                    r_citation_sentences.append(sentence)

        return r_citations, r_citation_sentences
    return [], []

def extract_element_boxes(pdf_path, get_citation=False):
    
    # Extract pages using PDFMiner
    pages = list(extract_pages(pdf_path))
    
    # Initialize list to store all text elements
    all_elements = []
    
    for page_num, page in enumerate(pages, start=1):
        
        # Get page dimensions
        pdf_width = page.width
        pdf_height = page.height
        
        # Iterate through layout objects on the page
        for element in page:
            if hasattr(element, 'get_text'): 
                # isinstance(element, (LTTextBoxHorizontal, LTTextBox, LTTextLine, LTChar)):
                # Get coordinates
                x0, y0, x1, y1 = element.bbox
                
                # Normalize coordinates
                x0_norm = x0 / pdf_width
                x1_norm = x1 / pdf_width
                y0_norm = y0 / pdf_height
                y1_norm = y1 / pdf_height
                
                # Add text with element type
                element_type = type(element).__name__
                
                # Extract text content
                text = ""
                text = element.get_text().strip()

                if is_reference_header(text):
                    continue

                element_info = {
                    "type": element_type,
                    "page": page_num,
                    "bbox": [x0_norm, y0_norm, x1_norm, y1_norm],
                    "text": text,
                }

                if get_citation:
                    citations, citation_sentences = extract_citation(text) 
                    if citations != []:                  
                        element_info["citations"] = citations
                        element_info["citation_sentences"] = citation_sentences       

                        all_elements.append(element_info)
    
    return all_elements
