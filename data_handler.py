from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import json
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

class NQForQfSDataset(Dataset):
    def __init__(self, data_path, data_loader_device):
        with open(data_path) as file:
            lines = file.readlines() # JSONL file
            self.dataset = [json.loads(line) for line in lines]
        self._size = len(self.dataset)
        self._sent_encoder = SentenceTransformer('all-mpnet-base-v2', device=data_loader_device) # emb-dim = 768
            
    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        # We will only use the first summary
        item = self.dataset[index]
        
        query = item['query']
        document = item['document']
        summary_meta_data = item['summaries'][0]
        
        summary_text = document[summary_meta_data['start_char_index']: summary_meta_data['end_char_index']]
        document_prologue = document[:summary_meta_data['start_char_index'] - 1] # 1 for the space
        document_epilogue = document[summary_meta_data['end_char_index'] + 1:] # 1 for the space
        
        summary_sents = sent_tokenize(summary_text)
        document_prologue_sents = sent_tokenize(document_prologue)
        document_epilogue_sents = sent_tokenize(document_epilogue)
        document_sents = document_prologue_sents + summary_sents + document_epilogue_sents
        
        labels = torch.cat((
            torch.zeros(len(document_prologue_sents)),
            torch.ones(len(summary_sents)),
            torch.zeros(len(document_epilogue_sents))
            ))
        
        query_embedding = self._sent_encoder.encode(query).reshape(1, -1) # (1, emb_dim)
        document_embedding = self._sent_encoder.encode(document_sents) # (#doc_sents, emb_dim)
        summary_embedding = self._sent_encoder.encode(summary_sents) # (#summary_sents, emb_dim)
        
        return query_embedding, document_embedding, summary_embedding, labels
        