import datasets
from tqdm import tqdm
from cleantext import clean
import emoji
import json
import re

NQ_DATA_DIR = "./nq_dataset"
SPLITS_FILE_PATH = "./sample_splits.json"
HTML_TAGS_FILE_PATH = "./html_tags.json"
EMAIL_REPLACE_PATTERN = "<email>"
URL_REPLACE_PATTERN = "<url>"
USERNAME_REPLACE_PATTERN = "<uid>"
IP_ADDR_REPLACE_PATTERN = "<ip_addr>"
HTML_REPLACE_PATTERN = ""

class CleanException(Exception):
    def __init__(self, message=None):
        self.message = message

def load_dataset_and_splits():
    dataset = datasets.load_dataset(path=NQ_DATA_DIR)

    with open(SPLITS_FILE_PATH) as file:
        sample_splits = json.load(file)
        
    return dataset, sample_splits

def get_html_regexes():
    with open(HTML_TAGS_FILE_PATH) as file:
        html_tags = json.load(file)
        
    def get_regexp_for_tags(html_tags):
        regexps = []
        for tag in html_tags:
            regex = '<[/]?' + f'[{tag[0].lower()}{tag[0].upper()}]' + tag[1:] + '>'
            regexps.append(regex)
            
        full_regex = '(' + '|'.join(regexps) + r'|<[/]?[Tt][dr] colspan[\s]?=[\s]?[\"\'][\d]{1,}[\"\']>' + ')'
        return full_regex
    
    return get_regexp_for_tags(html_tags)

def clean_to_ascii(text):
    special_symb_pattern = r"[™©®]"
    cleaned_text = re.sub(special_symb_pattern, "", text)
    cleaned_text = clean(cleaned_text, lower=False)
    cleaned_text = emoji.replace_emoji(cleaned_text, "")
   
    return cleaned_text

def replace_email(text):
    pattern = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?" 
    cleaned_text = re.sub(pattern, EMAIL_REPLACE_PATTERN, text)
    
    return cleaned_text

def replace_html_tags(text):
    pattern = get_html_regexes()
    cleaned_text = re.sub(pattern, HTML_REPLACE_PATTERN, text)
    cleaned_text = re.sub(r"(\s[\s]?|\t[\t]?)", " ", cleaned_text)
    
    return cleaned_text

def replace_username(text):
    pattern = r"[\s\n]?@[A-Za-z\.]+[\s\n]?"
    cleaned_text = re.sub(pattern, USERNAME_REPLACE_PATTERN, text)
    
    return cleaned_text

def replace_url(text):
    basic_pattern = r"http[s]?\:\/\/([A-Za-z]+)\.([A-Za-z]+)(\.(com|org|ac|in|co))?(\/[\w+\/]{0,})"
    non_trivial_pattern = r"([A-Za-z]+)\.([A-Za-z]+)(\.(com|org|ac|in|co))?(\/[\w+\/]{0,})"
    cleaned_text = re.sub(basic_pattern, URL_REPLACE_PATTERN, text)
    cleaned_text = re.sub(non_trivial_pattern, URL_REPLACE_PATTERN, cleaned_text)
    
    return cleaned_text

def replace_ip_addr(text):
    pattern = r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\/[0-9]|1[0-9]|2[0-9]|3[0-2])?"
    cleaned_text = re.sub(pattern, IP_ADDR_REPLACE_PATTERN, text)
    
    return cleaned_text

def clean_text_string(text):
    cleaned_text = clean_to_ascii(text)
    cleaned_text = replace_email(cleaned_text)
    cleaned_text = replace_username(cleaned_text)
    cleaned_text = replace_url(cleaned_text)
    cleaned_text = replace_ip_addr(cleaned_text)
    cleaned_text = replace_html_tags(cleaned_text)
        
    return cleaned_text

def clean_document_summary_and_reindex(document_tokens, summary_start, summary_end):
    document_prologue = document_tokens[:summary_start]
    summary_tokens = document_tokens[summary_start: summary_end]
    document_epilogue = document_tokens[summary_end:]
    
    document_prologue_text = ' '.join(document_prologue)
    document_prologue_cleaned_text = clean_text_string(document_prologue_text)
    
    summary_text = ' '.join(summary_tokens)
    summary_cleaned_text = clean_text_string(summary_text)
    
    document_epilogue_text = ' '.join(document_epilogue)
    document_epilogue_cleaned_text = clean_text_string(document_epilogue_text)
    
    summary_start_char_index = len(document_prologue_cleaned_text) + 1 # 1 for the space
    summary_end_char_index = summary_start_char_index + len(summary_cleaned_text)
    
    document_processed = document_prologue_cleaned_text + ' ' + summary_cleaned_text + ' ' + document_epilogue_cleaned_text
    if document_processed[summary_start_char_index: summary_end_char_index] != summary_cleaned_text: raise RuntimeError

    return document_processed, summary_start_char_index, summary_end_char_index
    
def create_dataset(data_split, indices):
    created_dataset = []
    
    for index in tqdm(indices):
        sample = data_split[index]
        created_data_item = {}
        
        created_data_item['query'] = sample['question']['text']
        created_data_item['summaries'] = []
        document_tokens = sample['document']['tokens']['token']
        
        ext_summaries = sample['annotations']['long_answer']
        
        for ext_summary in ext_summaries:
            summary_start = ext_summary['start_token']
            summary_end = ext_summary['end_token']
            
            try: document_processed, summary_start_char_index, summary_end_char_index = clean_document_summary_and_reindex(document_tokens, summary_start, summary_end)
            except RuntimeError as r: raise CleanException(f"Error at index: {index}")
            
            created_data_item['summaries'].append({
                'start_char_index': summary_start_char_index,
                'end_char_index': summary_end_char_index
            })
            
        created_data_item['document'] = document_processed
        
        created_dataset.append(created_data_item)
        
    return created_dataset

if __name__ == '__main__':
    dataset, sample_splits = load_dataset_and_splits()
    try:
        train_dataset = create_dataset(dataset['train'], sample_splits['train'])
    except CleanException as e:
        print(e.message)
        
    try:
        valid_dataset = create_dataset(dataset['train'], sample_splits['valid'])
    except CleanException as e:
        print(e.message)
        
    try:
        test_dataset = create_dataset(dataset['validation'], sample_splits['test'])
    except CleanException as e:
        print(e.message)
        