import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Dict, Tuple, Callable
from supervised_contrastive_loss import SupervisedContrastiveLoss
from transformers import BertModel, BertConfig

class DiffusionModel(nn.Module):
    """
        This class acts as the Diffusion Model. It helps in sampling during both forward and backward diffusion.
        The forward diffusion is a simple case, as DDPM (see Ho et al., 2020). The backward diffusion is based on
        DDIM (see Song et al., 2021; Denoising Diffusion Implicit Models). DDIM works as follows:
        
        q_posterior(x_{t-1} | x_t, x_0) is analytically computable. If we can use the Diffusion Backbone, such as UNet,
        or Transformer to predict x_0, we can use this to sample x_{t-1} from q_posterior(x_{t-1} | x_t, x_0). This is 
        exactly what Song et al. (2021) propose.
        
        We will use DDIM as it generates higher quality outputs in less time (10x - 50x reported in paper).
    """
    def __init__(self, beta_schedule: str, diffusion_steps: int, transformer_kwargs: Dict, training_kwargs: Dict):
        """
            Constructor for the Diffusion Model.

            :param beta_schedule: The name of the schedule to use for obtaining betas for different time steps.
            :param diffusion_steps: The total number of steps of diffusion
        """
        super().__init__()
        self._beta_schedule = beta_schedule
        self.diffusion_steps = diffusion_steps
        self.betas = torch.tensor(self._generate_betas_for_schedule(beta_schedule, diffusion_steps)) # (diffusion_steps,)
        self.alphas = 1 - self.betas # (diffusion_steps,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=-1) # (diffusion_steps,)
        
        self.alphas_cumprod_t_minus_1 = torch.cat((torch.tensor([1.0]), self.alphas_cumprod[:-1]))
        self.alphas_cumprod_t_plus_1 = torch.cat((self.alphas_cumprod[1:], torch.tensor([0.0])))
        
        assert self.alphas_cumprod_t_minus_1.size(0) == self.diffusion_steps
        assert self.alphas_cumprod_t_plus_1.size(0) == self.diffusion_steps
        
        self.posterior_variance = self.betas * ((1 - self.alphas_cumprod_t_minus_1) / (1 - self.alphas_cumprod)) # σ_t | (diffusion_steps,)
        
        self.backbone = TransformerLatentDiffuser(hf_backbone_name=transformer_kwargs['ldm-hf-model-name'])
        self.encoder = TransformerEncoder(hf_backbone_name=transformer_kwargs['enc-hf-backbone-name'])
        
        self.sc_loss_obj = SupervisedContrastiveLoss()
        
        self.contrastive_loss_weightage = training_kwargs['contrastive-loss-weight']
        self.matching_loss_weightage = training_kwargs['matching-loss-weight']
        self.generation_loss_weightage = training_kwargs['generation-loss-weight']

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
        alpha_cumprod_t = alpha_cumprod_t.reshape(alpha_cumprod_t.size(0), 1, 1) # (bsz, 1, 1)
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
        alpha_cumprod_t = alpha_cumprod_t.reshape(alpha_cumprod_t.size(0), 1, 1) # (bsz, 1, 1)
        alphas_cumprod_t_minus_1_at_t = self.alphas_cumprod_t_minus_1[t] # (bsz,)
        alphas_cumprod_t_minus_1_at_t = alphas_cumprod_t_minus_1_at_t.reshape(alphas_cumprod_t_minus_1_at_t.size(0), 1, 1) # (bsz, 1, 1)
        beta_t = self.betas[t] # (bsz,)
        beta_t = beta_t.reshape(beta_t.size(0), 1, 1)
        
        mu_t = torch.sqrt(alpha_cumprod_t) * ((1 - alphas_cumprod_t_minus_1_at_t) / (1 - alpha_cumprod_t)) * x_next + \
                torch.sqrt(alphas_cumprod_t_minus_1_at_t) * (beta_t / (1 - alpha_cumprod_t)) * x_start # (bsz, num_sents, emb_dim)
                
        std_t = torch.sqrt(self.posterior_variance)
        
        return {
            'mu_t-1': mu_t, # (bsz, num_sent, emb_dim)
            'std_t-1': std_t, # (bsz, num_sent, emb_dim)
        }
    
    def p_sample(self, x_next, t):
        """
            This method performs backward diffusion sampling. It works in two steps:
                1. First, compute x_0 using the Backbone, Transformer in this case.
                2. Next, compute x_{t-1} using the q_posterior.
                
            :param x_next: A batch of inputs at several time steps, represented by t | x_next.size() == (bsz, num_sents, emb_dim)
            :param t: A set of time steps to which forward diffusion has been performed | t.size() == (bsz,)
        """
        pred_x_start = self.backbone(x_next, t) # Predicts x_0
        q_posterior_output = self.q_posterior_sample(x_next, pred_x_start, t) # Gives the mean and std_dev for t-1
        
        q_posterior_output.update({'x_start_pred': pred_x_start})
        return q_posterior_output # The mu_t-1 acts as the x_{t-1}
    
    def train(self, qd_embeddings, s_embeddings, t, qd_attention_mask, s_attention_mask, num_qd_sents, contrastive_labels, matching_labels):
        """
            The training loop for the Diffusion Model. The train step first performs the forward diffusion to time-step t.
            Following that it performs the DDIM based backward diffusion, which uses the predicted x_0 from the backbone. 
            Loss would be employed on the generated x_{t-1} and predicted x_0, along with the matching and contrastive losses
            from the DiffuSum paper (Zhang et al., 2023).
        
            :param qd_embeddings: A batch of inputs specifying query + document embeddings at time step, t = 0 | x_start.size() == (bsz, num_qd_sents, emb_dim)
            :param s_embeddings: A batch of inputs specifying summary embeddings at time step, t = 0 | x_start.size() == (bsz, num_sents - n_qd_sents, emb_dim)
            :param t: A set of time steps to which we want to perform forward diffusion | t.size() == (bsz,)
            :param qd_attention_mask: A padding mask for query + document sentences input | qd_attention_mask.size() == (bsz, num_qd_sents)
            :param s_attention_mask: A padding mask for summary sentences input | s_attention_mask.size() == (bsz, num_sents - num_qd_sents)
            :param num_qd_sents: An integer specifying the number of sentences within num_sents that belong to query and document.
        """
         
        x_start_in = self.encoder(qd_embeddings, s_embeddings, qd_attention_mask, s_attention_mask) # H^{in} == H^{qd} || H^s | H^{qd} attended to only qd sents, H^s attended to only s sents
        
        noise = torch.randn_like(x_start_in)
        betas_0 = self.betas[torch.zeros_like(t)].reshape(t.size(0), 1, 1) # (bsz, 1, 1)
        x_start_init_sample = x_start_in + torch.sqrt(betas_0) * noise # (bsz, num_sents, emb_dim) | Applies some noise to both qd sents and summary sents
        
        q_sample_output = self.q_sample(x_start_init_sample, t)
        x_t = q_sample_output['x_t']
        noise_added = q_sample_output['epsilon_t']
        
        x_t[:, :num_qd_sents, :] = x_start_in[:, :num_qd_sents, :] # The input qd sents should not change, the diffuser needs to learn to condition on the virgin qd sents
        
        pred_x_start = self.backbone(x_t, t, torch.cat((qd_attention_mask, s_attention_mask), dim=1))
        pred_x_start[:, :num_qd_sents, :] = x_start_in[:, :num_qd_sents, :] # The input qd sents should not change, the diffuser needs to learn to condition on the virgin qd sents
        
        contrastive_loss = self._contrastive_loss(x_start_in, contrastive_labels)
        matching_loss = self._matching_loss(x_start_in, num_qd_sents, matching_labels)
        
        gen_quality_loss = self._gen_quality_loss(x_start_in, pred_x_start, num_qd_sents, s_attention_mask)
        
        loss = self.generation_loss_weightage * gen_quality_loss \
            +  self.matching_loss_weightage * matching_loss \
            + self.contrastive_loss_weightage * contrastive_loss
            
        return {'loss': loss, 'contrastive-loss': contrastive_loss, 'matching-loss': matching_loss, 'gen-quality-loss': gen_quality_loss}
   
    def _contrastive_loss(self, x_start_in, contrastive_labels):
        sc_loss = 0
        for _in, _label in zip(x_start_in, contrastive_labels):
            sc_loss += self.sc_loss_obj(_in, _label)
        return sc_loss / x_start_in.size(0)
    
    def _matching_loss(x_start_in, num_qd_sents, matching_labels):
        matching_logits = torch.bmm(x_start_in[:, num_qd_sents:, :], x_start_in[:, :num_qd_sents, :].permute(0, 2, 1))
        return nn.functional.cross_entropy(input=matching_logits.reshape(-1, matching_logits.size(-1)), target=matching_labels.reshape(-1), ingore_index=-100) # -100 signifies the padded positions in summary inputs
        
    def _gen_quality_loss(self, x_start, pred_x_start, num_qd_sents, s_attention_mask):
        # Calculates the MSE loss for the generations
        squared_error_sequence = torch.square(x_start[:, num_qd_sents:] - pred_x_start[:, num_qd_sents:]).sum(dim=-1) / x_start.size(-1) # (bsz, #summary_sents) | mean across the embedding dimension
        return torch.mean(torch.sum(squared_error_sequence * s_attention_mask, dim=-1) / torch.sum(s_attention_mask, dim=-1)) # first mean across seq_len accounting for padding, followed by mean across batch
    
    def forward(self, qd_embeddings, s_embeddings, t, qd_attention_mask, s_attention_mask, num_qd_sents, contrastive_labels, matching_labels):
        return self.train(qd_embeddings, s_embeddings, t, qd_attention_mask, s_attention_mask, num_qd_sents, contrastive_labels, matching_labels)
    
class TransformerEncoder(nn.Module):
    def __init__(self, hf_backbone_name, diffusion_emb_dim=None):
        """
            Generates the Transformer encoder to get contextual sentence representations

            :param hf_backbone_name: Specifies the Huggingface model code, such as `bert-base-cased`
            :param diffusion_emb_dim: Specifies the embedding dimension of the Latent Diffusion Model
        """
        self.encoder = BertModel.from_pretrained(hf_backbone_name)
        self._diffusion_emb_dim = diffusion_emb_dim
        if diffusion_emb_dim is not None:
            self.doc_transform = nn.Sequential(
                nn.Linear(self.encoder.config.hidden_size, diffusion_emb_dim) # Changes from encoder dimension to diffuser dimension
            )
    
    def forward(self, qd_embeddings, s_embeddings, qd_attention_mask, s_attention_mask):
        """
            Returns the contextual sentence embeddings for the query + document and sentence

            :param qd_embeddings: The embeddings for query and document sentences -- non-contextual embeddings from Sentence Transformers | qd_embeddings.size() == (bsz, qd_sents, emb_dim)
            :param s_embeddings: The embeddings for summary sentences -- non-contextual embeddings from Sentence Transformers | s_embeddings.size() == (bsz, s_sents, emb_dim)
            :param qd_attention_mask: The attention mask for query and document sentences | qd_attention_mask.size() == (bsz, qd_sents)
            :param s_attention_mask: The attention mask for query and document sentences | s_attention_mask.size() == (bsz, s_sents)
        """
        
        qd_embeddings = self.encoder(inputs_embeds=qd_embeddings, attention_mask=qd_attention_mask)['last_hidden_state']
        if self._diffusion_emb_dim is not None: qd_embeddings = self.doc_transform(qd_embeddings)
        qd_embeddings = nn.functional.normalize(qd_embeddings)
        
        s_embeddings = self.encoder(inputs_embeds=s_embeddings, attention_mask=s_attention_mask)['last_hidden_state']
        if self._diffusion_emb_dim is not None: s_embeddings = self.doc_transform(s_embeddings)
        s_embeddings = nn.functional.normalize(s_embeddings)
        
        return torch.cat((qd_embeddings, s_embeddings), dim=1) # Concat along sequence length dimension
    
class TransformerLatentDiffuser(nn.Module):
    def __init__(self, config):
        """
            Latent Diffusion Model for converting x_t to x_0

            :param config: The configuration for the Latent Diffusion Model, it is a HuggingFace BertConfig object.
        """
        self.diffuser = BertModel(config)
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
        t_embedding = self._timestep_embedding(t)
        x_t = x_t + t_embedding
        x_t = nn.functional.normalize(x_t)
        pred_x_start = self.diffuser(inputs_embeds=x_t, attention_mask=attention_mask)
        pred_x_start = nn.functional.normalize(pred_x_start)
        
        return pred_x_start
    
# Unit Testing 
if __name__ == '__main__':
    model = DiffusionModel('sqrt', 500)