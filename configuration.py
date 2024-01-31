import torch
from transformers import BartConfig, BertConfig

######--------CHANGEABLE VARS--------######

encoder_configs = {
    'bart': 'facebook/bart-base',
    'bert': 'bert-base-uncased',
    'scratch': {
        'hidden_size': 768,
        'num_hidden_layers': 8,
        'num_attention_heads': 8
    }
}

diffuser_config = {
    'bart': 'facebook/bart-base',
    'bert': 'bert-base-uncased',
    'scratch': {
        'hidden_size': 128,
        'num_hidden_layers': 12,
        'num_attention_heads': 16
    }
}

encoder_type = 'scratch'
diffuser_type = 'scratch'

device_data_loader = 'cuda'
model_type = 'diffusion' # One of `naive-baseline`, `att-flow`, `diffusion` and `improved-rl-diffusion`
hf_backbone_name = encoder_configs[encoder_type]
beta_schedule_name = 'cosine' # One of `linear` and `cosine`
diffusion_steps = 500
max_summary_len = 100
train_batch_size = 16
epochs = 10
grad_acc = 8
learning_rate = 5e-5
dtype = torch.float32
hf_backbone_name_ldm = diffuser_config[diffuser_type]

if diffuser_type == 'bart': ldm_config = BartConfig.from_pretrained(hf_backbone_name_ldm) # The max pos embeddings attribute gets dynamically changed
elif diffuser_type == 'bert': ldm_config = BertConfig.from_pretrained(hf_backbone_name_ldm) # The max pos embeddings attribute gets dynamically changed
elif diffuser_type == 'scratch': ldm_config = BertConfig(**hf_backbone_name_ldm) # Loads a new BertConfig

class Configuration:
    def __init__(self, run_name):
        ######--------CONFIGURATION STATE VARS--------######
        self.RUN_NAME = run_name
        
        ######--------DATA VARS--------######
        self.DATA_PATH = './nq-cleaned-dataset-extractive-qfs-v2'
        self.DATA_LOADER_DEVICE = device_data_loader
        self.OUTPUT_DIR = './run-files/{}'.format(run_name)
        self.MODEL_PRETRAINED_PATH = None
        
        ######--------MODEL VARS--------######
        self.MODEL_TYPE = model_type
        self.SAVED_DIR = None
        
        ######--------TRAINING VARS--------######
        self.TRAIN_BATCH_SIZE = train_batch_size
        self.EVAL_BATCH_SIZE = 16
        self.GRAD_ACC = grad_acc
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.USE_BF16 = (dtype == torch.bfloat16)
        self.USE_FP16 = ((dtype != torch.bfloat16) and (dtype == torch.float16))
        self.EVAL_STEPS = 500
        self.LR_SCHEDULER = 'cosine'
        self.LR_WARMUP = 1500
        self.OPTIMIZER_NAME = 'adam'
        self.WEIGHT_DECAY = 0.05
        self.MAX_GRAD_NORM = 1.0
        
        ######--------LOGGING VARS--------######
        self.LOG_STEPS = 5
    
    def load_model_args(self):
        if self.MODEL_TYPE == 'naive-baseline': return {'hf-backbone-name': hf_backbone_name}
        elif self.MODEL_TYPE == 'diffusion':
            return {
            'beta-schedule': beta_schedule_name,
            'diffusion-steps': diffusion_steps,
            'transformer-kwargs': {
                    'enc-hf-backbone-name': hf_backbone_name,
                    'ldm-hf-config': ldm_config,
                    'max-summary-len': max_summary_len,
                    'sbert-encoding-dim': 768,
                    'max-query-doc-len': 1024,
                    'test-sents': 10
                },
            'training-kwargs': {
                    'contrastive-loss-weight': 1e-1,
                    'matching-loss-weight': 1,
                    'generation-loss-weight': 100,
                    'regularization-loss-weightage': 1
                },
            'dtype': dtype
            }
        
    def serialize(self):
        pass
