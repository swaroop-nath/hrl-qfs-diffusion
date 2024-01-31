import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Callable
from supervised_contrastive_loss import SupervisedContrastiveLoss
from transformers import BartModel, BertModel, BertConfig, BartConfig
from utils import get_rouge_score
import os
from uuid import uuid1

class _BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._log_dict = {}
        
    def _update_logs(self, output_dict):
        for k, v in output_dict.items():
            output_dict[k] = v.unsqueeze(dim=0).reshape(-1)
        for k, v in output_dict.items():
            v = v.detach()
            existing_log = self._log_dict.get(k, None)
            if existing_log is None: self._log_dict[k] = v
            else: 
                new_log = torch.cat((existing_log, v.to(existing_log.device)), dim=-1).squeeze()
                self._log_dict[k] = new_log
    
    def get_logs(self):
        for k, v in self._log_dict.items():
            self._log_dict[k] = torch.mean(v).detach().item()
        return self._log_dict            
            
    def update_parameters_on_step_end(self):
        self._log_dict = {}
        
    def get_max_length(self):
        return self._max_len

class DiffusionModel(_BaseModel):
    """
        This class acts as the Diffusion Model. It helps in sampling during both forward and backward diffusion.
        The forward diffusion is a simple case, as DDPM (see Ho et al., 2020). The backward diffusion is based on
        DDIM (see Song et al., 2021; Denoising Diffusion Implicit Models). DDIM works as follows:
        
        q_posterior(x_{t-1} | x_t, x_0) is analytically computable. If we can use the Diffusion Backbone, such as UNet,
        or Transformer to predict x_0, we can use this to sample x_{t-1} from q_posterior(x_{t-1} | x_t, x_0). This is 
        exactly what Song et al. (2021) propose.
        
        We will use DDIM as it generates higher quality outputs in less time (10x - 50x reported in paper).
    """
    def __init__(self, beta_schedule: str, diffusion_steps: int, transformer_kwargs: Dict, training_kwargs: Dict, dtype: torch.typename = torch.float32):
        """
            Constructor for the Diffusion Model.

            :param beta_schedule: The name of the schedule to use for obtaining betas for different time steps. One of `linear` or `cosine`
            :param diffusion_steps: The total number of steps of diffusion
        """
        super().__init__()
        self._beta_schedule = beta_schedule
        self.diffusion_steps = diffusion_steps
        self.betas = torch.tensor(self._generate_betas_for_schedule(beta_schedule, diffusion_steps), dtype=dtype) # (diffusion_steps,)
        self.alphas = 1 - self.betas # (diffusion_steps,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=-1) # (diffusion_steps,)
        
        self.alphas_cumprod_t_minus_1 = torch.cat((torch.tensor([1.0]), self.alphas_cumprod[:-1]))
        self.alphas_cumprod_t_plus_1 = torch.cat((self.alphas_cumprod[1:], torch.tensor([0.0])))
        
        assert self.alphas_cumprod_t_minus_1.size(0) == self.diffusion_steps
        assert self.alphas_cumprod_t_plus_1.size(0) == self.diffusion_steps
        
        self.posterior_variance = self.betas * ((1 - self.alphas_cumprod_t_minus_1) / (1 - self.alphas_cumprod)) # σ_t | (diffusion_steps,)
        
        self.encoder = TransformerEncoder(hf_backbone_name=transformer_kwargs['enc-hf-backbone-name'], sent_emb_dim_in=transformer_kwargs['sbert-encoding-dim'], diffusion_emb_dim=transformer_kwargs['ldm-hf-config'].hidden_size, **transformer_kwargs)
        
        ldm_config = transformer_kwargs['ldm-hf-config']
        ldm_config.max_position_embeddings = transformer_kwargs['max-query-doc-len'] + transformer_kwargs['max-summary-len'] # Max length for the diffuser is set here
        self.backbone = TransformerLatentDiffuser(config=ldm_config)
        
        self.sc_loss_obj = SupervisedContrastiveLoss()
        
        self.contrastive_loss_weightage = training_kwargs['contrastive-loss-weight']
        self.matching_loss_weightage = training_kwargs['matching-loss-weight']
        self.generation_loss_weightage = training_kwargs['generation-loss-weight']
        self.regularization_loss_weightage = training_kwargs['regularization-loss-weightage']
        
        self._max_len = transformer_kwargs['max-query-doc-len'] # Use the encoder max pos, as Diffuser is not Pre-Trained, so that is not the bottleneck
        self._max_summary_len = transformer_kwargs['max-summary-len'] # This will help set the max pos embeddings for Diffuser
        self._num_test_sentences = transformer_kwargs['test-sents'] # The number of sentences to sample from the diffuser during testing

    def _generate_betas_for_schedule(self, beta_schedule: str, diffusion_steps: int) -> np.ndarray:
        """
            Get a pre-defined beta schedule for the given name.

            The beta schedule library consists of beta schedules which remain similar
            in the limit of num_diffusion_timesteps.
            Beta schedules may be added, but should not be removed or changed once
            they are committed to maintain backwards compatibility.
        """
        if beta_schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            A = 1000
            scale = A / diffusion_steps # For this diffusion steps has to be greater than or equal to A.
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(
                beta_start, beta_end, diffusion_steps, dtype=np.float64
            )
        elif beta_schedule == "cosine":
            return self._betas_for_alpha_bar(
                diffusion_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {beta_schedule}")
        
    def _betas_for_alpha_bar(self, num_diffusion_timesteps: int, alpha_bar: Callable[[int], int], max_beta: str=0.999) -> np.ndarray:
        """
            Create a beta schedule that discretizes the given alpha_t_bar function,
            which defines the cumulative product of (1-beta) over time from t = [0,1].

            :param num_diffusion_timesteps: the number of betas to produce.
            :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                            produces the cumulative product of (1-beta) up to that
                            part of the diffusion process.
            :param max_beta: the maximum beta to use; use values lower than 1 to
                            prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
            This method performs the q(x_t | x_{t-1}) sampling. By reparametrization trick, it helps in computing 
            q(x_t | x_0).
            
            :param x_start: A batch of inputs at time step, t = 0 | x_start.size() == (bsz, num_sents, emb_dim)
            :param t: A set of time steps to which we want to perform forward diffusion | t.size() == (bsz,)
        """
        alpha_cumprod_t = self.alphas_cumprod[t] # (bsz,)
        alpha_cumprod_t = alpha_cumprod_t.reshape(alpha_cumprod_t.size(0), 1, 1).to(x_start.device) # (bsz, 1, 1)
        noise = torch.randn_like(x_start) # noise.size() == (bsz, num_sents, emb_dim)
        mu_t = torch.sqrt(alpha_cumprod_t) * x_start # mu_t.size() == (bsz, num_sents, emb_dim)
        std_t = torch.sqrt(1 - alpha_cumprod_t) # std_t.size() == (bsz, 1, 1) | The cov matrix is an eye matrix, with size (emb_dim, emb_dim)
        
        x_t = mu_t + std_t * noise # (bsz, num_sents, emb_dim)
        
        return {
            'mu_t': mu_t, # (bsz, num_sents, emb_dim)
            'std_t': std_t, # (bsz, 1, 1)
            'x_t': x_t, # (bsz, num_sents, emb_dim)
            'epsilon_t': noise, # (bsz, num_sents, emb_dim)
        }
    
    def q_posterior_sample(self, x_next, x_start, t):
        """
            This method performs the q_posterior(x_{t-1} | x_t, x_0). This is analytically computable.
            
            :param x_next: A batch of inputs at several time steps, represented by t | x_next.size() == (bsz, num_sents, emb_dim)
            :param x_start: Same batch of inputs at time step, t = 0 | x_start.size() == (bsz, num_sents, emb_dim)
            :param t: A set of time steps to which forward diffusion has been performed | t.size() == (bsz,)
        """
        alpha_cumprod_t = self.alphas_cumprod[t] # (bsz,)
        alpha_cumprod_t = alpha_cumprod_t.reshape(alpha_cumprod_t.size(0), 1, 1).to(x_start.device) # (bsz, 1, 1)
        alphas_cumprod_t_minus_1_at_t = self.alphas_cumprod_t_minus_1[t] # (bsz,)
        alphas_cumprod_t_minus_1_at_t = alphas_cumprod_t_minus_1_at_t.reshape(alphas_cumprod_t_minus_1_at_t.size(0), 1, 1).to(x_start.device) # (bsz, 1, 1)
        beta_t = self.betas[t] # (bsz,)
        beta_t = beta_t.reshape(beta_t.size(0), 1, 1).to(x_start.device)
        
        mu_t = torch.sqrt(alpha_cumprod_t) * ((1 - alphas_cumprod_t_minus_1_at_t) / (1 - alpha_cumprod_t)) * x_next + \
                torch.sqrt(alphas_cumprod_t_minus_1_at_t) * (beta_t / (1 - alpha_cumprod_t)) * x_start # (bsz, num_sents, emb_dim)
                
        std_t = torch.sqrt(self.posterior_variance)
        
        return {
            'mu_t-1': mu_t, # (bsz, num_sent, emb_dim)
            'std_t-1': std_t, # (bsz, num_sent, emb_dim)
        }
    
    @torch.no_grad()
    def p_sample(self, x_next, t, attn_mask):
        """
            This method performs backward diffusion sampling. It works in two steps:
                1. First, compute x_0 using the Backbone, Transformer in this case.
                2. Next, compute x_{t-1} using the q_posterior.
                
            :param x_next: A batch of inputs at several time steps, represented by t | x_next.size() == (bsz, num_sents, emb_dim)
            :param t: A set of time steps to which forward diffusion has been performed | t.size() == (bsz,)
        """
        pred_x_start = self.backbone(x_next, t, attn_mask) # Predicts x_0
        q_posterior_output = self.q_posterior_sample(x_next, pred_x_start, t) # Gives the mean and std_dev for t-1
        
        q_posterior_output.update({'x_start_pred': pred_x_start})
        return q_posterior_output # The mu_t-1 acts as the x_{t-1}
    
    @torch.no_grad()
    def p_sample_loop(self, qd_embeddings, qd_attention_mask):
        noise = torch.randn((qd_embeddings.size(0), self._num_test_sentences, qd_embeddings.size(-1)), device=qd_embeddings.device) # (bsz, #summar_sents, emb_dim)
        x_next = torch.cat((qd_embeddings, noise), dim=1) 
        attn_mask = torch.cat((qd_attention_mask, torch.ones(noise.size(0), noise.size(1), device=noise.device)), dim=1) # (bsz, #qd_sents + #summary_sents, emb_dim)
        
        for step in range(self.diffusion_steps - 1, -1, -1):
            time_step = torch.ones((qd_embeddings.size(0)), dtype=torch.long, device=qd_embeddings.device) * step
            p_sample_output = self.p_sample(x_next, time_step, attn_mask)
            x_next_qds = p_sample_output['mu_t-1']
            x_next_summary = x_next_qds[:, qd_embeddings.size(1):, :]
            x_next = torch.cat((qd_embeddings, x_next_summary), dim=1)
            
        return x_next[:, qd_embeddings.size(1):, :] # (bsz, #summary_sents, emb_dim)
    
    def get_max_summary_len(self):
        return self._max_summary_len
    
    @torch.no_grad()
    def generate(self, **batch):
        qd_embeddings = batch['qd_embeddings']
        qd_attention_mask = batch['qd_attention_mask']
        
        x_start_in = self.encoder(qd_embeddings, qd_attention_mask)
        
        noise = torch.randn_like(x_start_in)
        betas_0 = self.betas[torch.zeros(qd_embeddings.size(0), dtype=torch.long)].reshape(qd_embeddings.size(0), 1, 1).to(x_start_in.device) # (bsz, 1, 1)
        x_start_init_sample = x_start_in + torch.sqrt(betas_0) * noise # (bsz, num_sents, emb_dim) | Applies some noise to both qd sents and summary sents
        
        summary_embeddings = self.p_sample_loop(x_start_init_sample, qd_attention_mask)
        d_embeddings = x_start_in[:, 1:, :] # first the query | (bsz, #document_sents, emb_dim)
        d_mask = qd_attention_mask[:, 1:] # first the query | (bsz, #document_sents)
        d_mask = d_mask.unsqueeze(dim=1)
        
        logits = torch.bmm(summary_embeddings, d_embeddings.permute(0, 2, 1)) # (bsz, #summary_sents, #document_sents)
        logits = logits * d_mask + (-torch.tensor(float('inf'))) * (1 - d_mask) # (bsz, #summary_sents, #document_sents) | Masked regions are -inf
        proba = torch.softmax(logits, dim=-1)
        batch_doc_winners = torch.argmax(proba, dim=-1) # (bsz, #summary_sents)
        
        batch_document_sents = batch['document_text_sents'] # List of documents sentences --> each element is a list
        batch_summary = []
        for batch_idx, doc_winner in enumerate(batch_doc_winners):
            document_sents = batch_document_sents[batch_idx]
            summary_sents = [document_sents[winner_index.cpu().item()] for winner_index in doc_winner]
            batch_summary.append(' '.join(summary_sents))
        
        return batch_summary
    
    def train_supervised(self, qd_embeddings, s_embeddings, t, qd_attention_mask, s_attention_mask, num_qd_sents, contrastive_labels, matching_labels):
        """
            The training loop for the Diffusion Model. The train step first performs the forward diffusion to time-step t.
            Following that it performs the DDIM based backward diffusion, which uses the predicted x_0 from the backbone. 
            Loss would be employed on the generated x_{t-1} and predicted x_0, along with the matching and contrastive losses
            from the DiffuSum paper (Zhang et al., 2023).
        
            :param qd_embeddings: A batch of inputs specifying query + document embeddings at time step, t = 0 | x_start.size() == (bsz, num_qd_sents, emb_dim)
            :param s_embeddings: A batch of inputs specifying summary embeddings at time step, t = 0 | x_start.size() == (bsz, num_sents - num_qd_sents, emb_dim)
            :param t: A set of time steps to which we want to perform forward diffusion | t.size() == (bsz,)
            :param qd_attention_mask: A padding mask for query + document sentences input | qd_attention_mask.size() == (bsz, num_qd_sents)
            :param s_attention_mask: A padding mask for summary sentences input | s_attention_mask.size() == (bsz, num_sents - num_qd_sents)
            :param num_qd_sents: An integer specifying the number of sentences within num_sents that belong to query and document (max).
            :param contrastive_labels: Contrastive labels as per https://arxiv.org/pdf/2305.01735.pdf | (bsz, num_qd_sents + num_s_sents)
            :param matching_labels: A tensor of (bsz, num_sents - num_qd_sents) dimension | Each (i, j) --> k entry suggests that for the ith instance in the batch, the jth sentence of the summary is the kth sentence of the document
        """
         
        x_start_in = self.encoder(qd_embeddings, qd_attention_mask, s_embeddings, s_attention_mask) # H^{in} == H^{qd} || H^s | H^{qd} attended to only qd sents, H^s attended to only s sents
        
        noise = torch.randn_like(x_start_in)
        betas_0 = self.betas[torch.zeros_like(t)].reshape(t.size(0), 1, 1).to(x_start_in.device) # (bsz, 1, 1)
        x_start_init_sample = x_start_in + torch.sqrt(betas_0) * noise # (bsz, num_sents, emb_dim) | Applies some noise to both qd sents and summary sents
        
        q_sample_output = self.q_sample(x_start_init_sample, t)
        x_t = q_sample_output['x_t']
        noise_added = q_sample_output['epsilon_t']
        
        x_t[:, :num_qd_sents, :] = x_start_in[:, :num_qd_sents, :] # The input qd sents should not change, the diffuser needs to learn to condition on the virgin qd sents
        
        qds_mask = torch.cat((qd_attention_mask, s_attention_mask), dim=1)
        pred_x_start = self.backbone(x_t, t, qds_mask)
        pred_x_start[:, :num_qd_sents, :] = x_start_in[:, :num_qd_sents, :] # The input qd sents should not change, the diffuser needs to learn to condition on the virgin qd sents
        
        contrastive_loss = self._contrastive_loss(x_start_in, contrastive_labels)
        matching_loss = self._matching_loss(x_start_in, num_qd_sents, matching_labels)
        
        gen_quality_loss = self._gen_quality_loss(x_start_in, pred_x_start, num_qd_sents, s_attention_mask)
        
        regularization_loss = torch.mean(((torch.square(x_t).sum(dim=-1) / x_t.size(-1)) * qds_mask).sum(dim=-1) / qds_mask.size(-1))
        
        loss = self.generation_loss_weightage * gen_quality_loss \
            + self.matching_loss_weightage * matching_loss \
            + self.contrastive_loss_weightage * contrastive_loss \
            + self.regularization_loss_weightage * regularization_loss
            
        output_dict = {'loss': loss, 'contrastive-loss': contrastive_loss, 'matching-loss': matching_loss, 'gen-quality-loss': gen_quality_loss, 'regularization-loss': regularization_loss}
        self._update_logs(output_dict)
        
        return {'loss': loss, 'contrastive-loss': contrastive_loss, 'matching-loss': matching_loss, 'gen-quality-loss': gen_quality_loss, 'regularization-loss': regularization_loss}
   
    def _contrastive_loss(self, x_start_in, contrastive_labels):
        sc_loss = 0
        for _in, _label in zip(x_start_in, contrastive_labels):
            sc_loss += self.sc_loss_obj(_in, _label)
        return sc_loss / x_start_in.size(0)
    
    def _unsup_contrastive_loss(self, x_start_in, num_qd_sents, qd_attention_mask, s_attention_mask):
        # x_start_in.size() == (bsz, num_qd_sents + num_s_sents, emb_dim)
        qd_embedding = x_start_in[:, :num_qd_sents]
        s_embedding = x_start_in[:, num_qd_sents:]
        
        qd_labels = torch.arange(num_qd_sents).expand(qd_attention_mask.size()).to(qd_attention_mask.device)
        qd_labels = qd_labels * qd_attention_mask + (1 - qd_attention_mask) * -100 # This makes padded place labels as -100
        
        s_labels = torch.arange(s_embedding.size(1)).expand(s_attention_mask.size()).to(s_attention_mask.device)
        s_labels = s_labels * s_attention_mask + (1 - s_attention_mask) * -100 # This makes padded place labels as -100
        
        qd_logits = torch.bmm(qd_embedding, qd_embedding.permute(0, 2, 1)) # (bsz, num_qd_sents, num_qd_sents)
        qd_loss = nn.functional.cross_entropy(input=qd_logits.reshape(-1, qd_logits.size(-1)), target=qd_labels.reshape(-1).to(torch.long))
        
        s_logits = torch.bmm(s_embedding, s_embedding.permute(0, 2, 1)) # (bsz, num_s_sents, num_s_sents)
        s_loss = nn.functional.cross_entropy(input=s_logits.reshape(-1, s_logits.size(-1)), target=s_labels.reshape(-1).to(torch.long))
        return qd_loss + s_loss
    
    def _matching_loss(self, x_start_in, num_qd_sents, matching_labels):
        matching_logits = torch.bmm(x_start_in[:, num_qd_sents:, :], x_start_in[:, :num_qd_sents, :].permute(0, 2, 1)) # (bsz, num_s_sents, num_qd_sents)
        return nn.functional.cross_entropy(input=matching_logits.reshape(-1, matching_logits.size(-1)), target=matching_labels.reshape(-1), ignore_index=-100) # -100 signifies the padded positions in summary inputs
        
    def _gen_quality_loss(self, x_start, pred_x_start, num_qd_sents, s_attention_mask):
        # Calculates the MSE loss for the generations
        squared_error_sequence = torch.square(x_start[:, num_qd_sents:] - pred_x_start[:, num_qd_sents:]).sum(dim=-1) / x_start.size(-1) # (bsz, #summary_sents) | mean across the embedding dimension
        return torch.mean(torch.sum(squared_error_sequence * s_attention_mask, dim=-1) / torch.sum(s_attention_mask, dim=-1)) # first mean across seq_len accounting for padding, followed by mean across batch
    
    def _sample_timesteps(self, bsz):
        time_steps = torch.randint(low=0, high=self.diffusion_steps, size=(bsz,))
        return time_steps
    
    def forward(self, **batch):
        qd_embeddings, s_embeddings, qd_attention_mask, s_attention_mask, num_qd_sents, contrastive_labels, matching_labels = \
            batch['qd_embeddings'], batch['s_embeddings'], batch['qd_attention_mask'], batch['s_attention_mask'], batch['num_qd_sents'], \
            batch['contrastive_labels'], batch['matching_labels']
        t = self._sample_timesteps(qd_embeddings.size(0))
        return self.train_supervised(qd_embeddings, s_embeddings, t, qd_attention_mask, s_attention_mask, num_qd_sents, contrastive_labels, matching_labels)
    
class TransformerEncoder(nn.Module):
    def __init__(self, hf_backbone_name, sent_emb_dim_in=None, diffusion_emb_dim=None, **kwargs):
        """
            Generates the Transformer encoder to get contextual sentence representations

            :param hf_backbone_name: Specifies the Huggingface model code, such as `bert-base-cased`
            :param diffusion_emb_dim: Specifies the embedding dimension of the Latent Diffusion Model
        """
        super().__init__()
        if 'bart' in hf_backbone_name: self.encoder = BartModel.from_pretrained(hf_backbone_name).encoder
        elif 'bert' in hf_backbone_name: self.encoder = BertModel.from_pretrained(hf_backbone_name)
        elif type(hf_backbone_name) == dict:
            config = BertConfig(**hf_backbone_name)
            config.max_position_embeddings = kwargs['max-query-doc-len'] + kwargs['max-summary-len']
            self.encoder = BertModel(config)
        self.config = self.encoder.config
        self._sent_emb_dim_in = sent_emb_dim_in
        self._diffusion_emb_dim = diffusion_emb_dim
        if sent_emb_dim_in is not None:
            self.doc_transform_in = nn.Sequential(
                nn.Linear(sent_emb_dim_in, self.config.hidden_size) # Changes from sbert dimension to encoder dimension
            )
        if diffusion_emb_dim is not None:
            self.doc_transform_out = nn.Sequential(
                nn.Linear(self.encoder.config.hidden_size, diffusion_emb_dim) # Changes from encoder dimension to diffuser dimension
            )
    
    def forward(self, qd_embeddings, qd_attention_mask, s_embeddings=None, s_attention_mask=None):
        """
            Returns the contextual sentence embeddings for the query + document and sentence

            :param qd_embeddings: The embeddings for query and document sentences -- non-contextual embeddings from Sentence Transformers | qd_embeddings.size() == (bsz, qd_sents, emb_dim)
            :param s_embeddings: The embeddings for summary sentences -- non-contextual embeddings from Sentence Transformers | s_embeddings.size() == (bsz, s_sents, emb_dim)
            :param qd_attention_mask: The attention mask for query and document sentences | qd_attention_mask.size() == (bsz, qd_sents)
            :param s_attention_mask: The attention mask for query and document sentences | s_attention_mask.size() == (bsz, s_sents)
        """
        
        if self._sent_emb_dim_in is not None: qd_embeddings = self.doc_transform_in(qd_embeddings)
        qd_embeddings = nn.functional.normalize(qd_embeddings, dim=2)
        qd_embeddings_c = self.encoder(inputs_embeds=qd_embeddings, attention_mask=qd_attention_mask)['last_hidden_state']
        qd_embeddings_c = nn.functional.normalize(qd_embeddings_c, dim=2)
        if self._diffusion_emb_dim is not None: qd_embeddings_c = self.doc_transform_out(qd_embeddings_c)
        # qd_embeddings_c = nn.functional.normalize(qd_embeddings_c, dim=2)
        
        if s_embeddings is None: return qd_embeddings_c
        
        if self._sent_emb_dim_in is not None: s_embeddings = self.doc_transform_in(s_embeddings)
        s_embeddings = nn.functional.normalize(s_embeddings, dim=2)
        # qds_embeddings = torch.cat((qd_embeddings, s_embeddings), dim=1)
        # qds_mask = torch.cat((qd_attention_mask, s_attention_mask), dim=1)
        s_embeddings_c = self.encoder(inputs_embeds=s_embeddings, attention_mask=s_attention_mask)['last_hidden_state']
        s_embeddings_c = nn.functional.normalize(s_embeddings_c, dim=2)
        if self._diffusion_emb_dim is not None: s_embeddings_c = self.doc_transform_out(s_embeddings_c)
        # s_embeddings_c = nn.functional.normalize(s_embeddings_c, dim=2)
        
        return torch.cat((qd_embeddings_c, s_embeddings_c), dim=1) # Concat along sequence length dimension
    
class TransformerLatentDiffuser(nn.Module):
    def __init__(self, config):
        """
            Latent Diffusion Model for converting x_t to x_0

            :param config: The configuration for the Latent Diffusion Model, it is a HuggingFace BertConfig object.
        """
        super().__init__()
        if isinstance(config, BartConfig): self.diffuser = BartModel(config).encoder
        elif isinstance(config, BertConfig): self.diffuser = BertModel(config)
        self.config = config
    
    def _timestep_embedding(self, timesteps, max_period=10000):
        dim = self.config.hidden_size
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, x_t, t, attention_mask):
        t_embedding = self._timestep_embedding(t).to(x_t.device) # (bsz, emb_dim)
        t_embedding = t_embedding.reshape(t_embedding.size(0), 1, t_embedding.size(1)) # (bsz, 1, emb_dim)
        x_t = x_t + t_embedding
        x_t = nn.functional.normalize(x_t, dim=2)
        pred_x_start = self.diffuser(inputs_embeds=x_t, attention_mask=attention_mask)['last_hidden_state']
        pred_x_start = nn.functional.normalize(pred_x_start, dim=2)
        
        return pred_x_start
    
class NaiveBaseline(_BaseModel):
    def __init__(self, hf_backbone_name):
        super().__init__()
        self.encoder_model = BartModel.from_pretrained(hf_backbone_name).encoder
        self.config = self.encoder_model.config
        self._max_len = self.config.max_position_embeddings
        num_output_classes_start = 1 # 1 --> either start or not
        num_output_classes_end = 1 # 1 --> either start or not
        self._classifier_start_ff = nn.Linear(in_features=self.config.hidden_size, out_features=num_output_classes_start)
        self._classifier_end_ff = nn.Linear(in_features=self.config.hidden_size, out_features=num_output_classes_end)
        
    def model_forward_pass(self, qd_embeddings, qd_attention_mask):
        model_outputs = self.encoder_model(inputs_embeds=qd_embeddings, attention_mask=qd_attention_mask)['last_hidden_state'] # (bsz, qd_seq_len, emb_dim)
        model_outputs = nn.functional.normalize(model_outputs, dim=2, p=2)
        contextual_doc_embeddings = model_outputs[:, 1:, :] # The first embedding is for query
        
        start_logits = self._classifier_start_ff(contextual_doc_embeddings)\
            .reshape(contextual_doc_embeddings.size(0), contextual_doc_embeddings.size(1)) # (bsz, seq_len, emb_dim) --> (bsz, seq_len, 1) --> (bsz, seq_len).
            
        end_logits = self._classifier_end_ff(contextual_doc_embeddings)\
            .reshape(contextual_doc_embeddings.size(0), contextual_doc_embeddings.size(1)) # (bsz, seq_len, emb_dim) --> (bsz, seq_len, 1) --> (bsz, seq_len).
        
        return start_logits, end_logits
        
    def forward(self, **batch):
        qd_embeddings, qd_attention_mask, start_labels, end_labels = batch['qd_embeddings'], batch['qd_attention_mask'], batch['start_labels'], batch['end_labels']
        start_logits, end_logits = self.model_forward_pass(qd_embeddings, qd_attention_mask)
            
        start_prob_loss = nn.functional.cross_entropy(input=start_logits, target=start_labels)
        end_prob_loss = nn.functional.cross_entropy(input=end_logits, target=end_labels)
        
        output_dict = {
            'loss': start_prob_loss + end_prob_loss,
            'start-prob-loss': start_prob_loss,
            'end-prob-loss': end_prob_loss
        }
        
        self._update_logs(output_dict)
        
        return {
            'loss': start_prob_loss + end_prob_loss,
            'start-prob-loss': start_prob_loss,
            'end-prob-loss': end_prob_loss
        }
        
    @torch.no_grad()
    def generate(self, **batch):
        document_sents = batch['document_text_sents'] # List of documents sentences --> each element is a list
        gt_summaries_text = batch['summary_text'] # List of summaries --> each element is a string
        pred_summaries_text = []
        
        qd_embeddings, qd_attention_mask = batch['qd_embeddings'], batch['qd_attention_mask']
        # start_logits.size() == end_logits.size() == (bsz, seq_len_doc)
        start_logits, end_logits = self.model_forward_pass(qd_embeddings, qd_attention_mask)
        
        for batch_index in range(len(document_sents)):
            start_index = torch.argmax(start_logits[batch_index, :len(document_sents[batch_index])]) # (,) --> an element
            end_index_offset = torch.argmax(end_logits[batch_index, start_index:len(document_sents[batch_index])], dim=-1).item() # (,) --> an element
            
            pred_summary = document_sents[batch_index][start_index : start_index + end_index_offset + 1]
            pred_summary_text = ' '.join(pred_summary)
            pred_summaries_text.append(pred_summary_text)
            
        return pred_summaries_text
