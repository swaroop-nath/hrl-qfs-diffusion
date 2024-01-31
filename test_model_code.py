from model_code import *

# Unit Testing
def test_transformer_encoder():
    import torch
    model = TransformerEncoder('bert-base-cased')
    
    d_model = model.encoder.config.hidden_size
    qd_embeddings = torch.randn((2, 5, d_model))
    s_embeddings = torch.randn((2, 3, d_model))
    qd_attention_mask = torch.randint(0, 2, size=(2, 5))
    s_attention_mask = torch.randint(0, 2, size=(2, 3))
    
    output = model(qd_embeddings, s_embeddings, qd_attention_mask, s_attention_mask) 
    assert output is not None
    assert output.size() == (2, 8, d_model)
    
def test_transformer_diffuser():
    import torch
    from transformers import BartConfig
    
    config = BartConfig.from_pretrained("facebook/bart-large")
    ldm = TransformerLatentDiffuser(config)
    
    x_t = torch.randn((2, 5, 128)) # (bsz, seq_len, emb_dim)
    t = torch.randint(low=0, high=400, size=(2,)) # (bsz,)
    attention_mask = torch.randint(low=0, high=2, size=(2, 5)) # (bsz, seq_len)
    
    output = ldm(x_t, t, attention_mask)
    assert output is not None
    assert output.size() == (2, 5, 128)
    
def test_diffusion_model(device='cpu'):
    import torch
    from transformers import BartConfig
    
    d_model = 128
    config = BartConfig.from_pretrained("facebook/bart-base")
    diffusion_model = DiffusionModel('cosine', 500, {'ldm-hf-config': config, 'enc-hf-backbone-name': 'facebook/bart-base', 'max-summary-len': 100}, \
            {'contrastive-loss-weight': 1e-3, 'matching-loss-weight': 1, 'generation-loss-weight': 1}).to(device)
    
    encoder_hidden_size = diffusion_model.encoder.config.hidden_size
    ldm_hidden_size = d_model
    
    bsz = 32
    qd_seq_len = 1024
    s_seq_len = 50
    
    qd_embeddings = torch.randn((bsz, qd_seq_len, encoder_hidden_size)).to(device) # (bsz, qd_sents, encoder_hidden_size)
    s_embeddings = torch.randn((bsz, s_seq_len, encoder_hidden_size)).to(device) # (bsz, s_sents, encoder_hidden_size)
    qd_attention_mask = torch.randint(low=0, high=2, size=(bsz, qd_seq_len)).to(device)
    s_attention_mask = torch.randint(low=0, high=2, size=(bsz, s_seq_len)).to(device)
    num_qd_sents = qd_seq_len
    contrastive_labels = torch.randint(low=0, high=1, size=(bsz, qd_seq_len + s_seq_len)).to(device)
    matching_labels = torch.randint(low=0, high=qd_seq_len, size=(bsz, s_seq_len)).to(device)
        
    output = diffusion_model(qd_embeddings, s_embeddings, qd_attention_mask, s_attention_mask, num_qd_sents, contrastive_labels, matching_labels)
    
    assert output is not None
 
if __name__ == '__main__':
    from tqdm import trange
    for _ in trange(50):
        test_diffusion_model(device='cuda:0')