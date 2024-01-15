from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
            )).to(torch.int) # (#doc_sents,)
        
        query_embedding = self._sent_encoder.encode(query, convert_to_tensor=True).reshape(1, -1).cpu() # (1, emb_dim) | TODO: IS ASSUMING ONE SENT PER QUERY OK? YES!
        document_embedding = self._sent_encoder.encode(document_sents, convert_to_tensor=True).cpu() # (#doc_sents, emb_dim)
        summary_embedding = self._sent_encoder.encode(summary_sents, convert_to_tensor=True).cpu() # (#summary_sents, emb_dim)
        
        return query_embedding, document_embedding, summary_embedding, labels
    
def _get_first_non_zero(temp_tensor):
    # temp_tensor.size() == (bsz, seq_len)
    rev_index = torch.arange(start=temp_tensor.size(1), end=0, step=-1)
    temp_2 = temp_tensor * rev_index
    
    return torch.argmax(temp_2, dim=1) # (bsz,)

def _get_batch_items(batch):
    query_embedding, document_embedding, summary_embedding, labels = tuple(zip(*batch))
    return list(query_embedding), list(document_embedding), list(summary_embedding), list(labels)
    
def _get_attention_mask(batched_embeddings):
    return (1 - torch.mean(torch.eq(batched_embeddings, -100).to(torch.float), dim=-1)).to(torch.int)

def _get_start_and_end(labels_tensor):
    # labels_tensor.size() == (bsz, #doc_sents)
    start = _get_first_non_zero(labels_tensor) # (bsz,)
    end = labels_tensor.size(1) - _get_first_non_zero(torch.flip(labels_tensor, dims=(1,))) - 1 # (bsz,)
    
    return start, end

def data_collator_for_naive_baseline(batch):
    # batch is a 4 tuple:
    # query_embedding --> (1, emb_dim) for the query sentence
    # document_embedding --> (#doc_sents, emb_dim) for the document sentences
    # summary_embedding --> (#summary_sents, emb_dim) for the summary sentences
    # labels --> (#doc_sents) --> 0 for non-summary sents, 1 for summary sent | 1's are contiguous
    
    query_embeddings, document_embeddings, _, labels = _get_batch_items(batch)

    query_embeddings = torch.stack(query_embeddings) # (bsz, emb_dim)
    document_embeddings = pad_sequence(document_embeddings, batch_first=True, padding_value=-100)
    d_attention_mask = _get_attention_mask(document_embeddings)
    
    labels = pad_sequence(labels, batch_first=True, padding_value=-100) # (bsz, #doc_sents)
    start_tensor_labels, end_tensor_labels = _get_start_and_end(labels)
    
    qd_embeddings = torch.cat((query_embeddings, document_embeddings), dim=1) # Along seq_len
    qd_attention_mask = torch.cat((torch.ones(query_embeddings.size(0), 1), d_attention_mask), dim=1) # Along seq_len
    
    assert (end_tensor_labels > start_tensor_labels).all(), "End is before Start"
    
    return {
        'qd_embeddings': qd_embeddings,
        'qd_attention_mask': qd_attention_mask,
        'start_labels': start_tensor_labels,
        'end_labels': end_tensor_labels
    }