from model_code import DiffusionModel, NaiveBaseline
from configuration import Configuration
from transformers import Trainer, TrainingArguments, TrainerCallback
from coolname import generate_slug
import wandb
import os
from tqdm import tqdm
from data_handler import data_collator_for_naive_baseline, data_collator_for_diffusum_baseline, NQForQfSDatasetV2
import numpy as np
import torch

##=========Custom Callback for Logging Setup=========##
class CustomTrainerCallback(TrainerCallback):
    def add_model(self, model):
        self._model = model
        
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        
    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        
    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        logs = self._model.get_logs()
        wandb_logs = {}
        for k, v in logs.items():
            if k not in ['loss']: wandb_logs['train/' + k] = v
        wandb.log(wandb_logs)
        self._model.update_parameters_on_step_end()
        
    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        logs = self._model.get_logs()
        wandb_logs = {}
        for k, v in logs.items():
            wandb_logs['eval/' + k] = v
        wandb.log(wandb_logs)
        self._model.update_parameters_on_step_end()
        
##=========Custom Trainer for Logging Setup=========##
class CustomTrainer(Trainer):
    def add_callback(self, callback):
        super().add_callback(callback)
        if isinstance(callback, CustomTrainerCallback): callback.add_model(self.model)
        
    def add_run_name(self, run_name):
        self.run_name = run_name
        
    def add_test_dataset(self, test_dataset):
        self.test_data_loader = self.get_test_dataloader(test_dataset)
        
    def run_on_test_dataset(self):
        pbar = tqdm(total=len(self.test_data_loader), desc='Running on test dataset')
        if not os.path.exists(f'test-output-dir/{self.run_name}'): os.makedirs(f'test-output-dir/{self.run_name}')
        for step, batch in enumerate(self.test_data_loader):
            unique_ids = batch.pop('unique_ids')
            query_texts = batch.pop('query_text')
            batch = self._prepare_inputs(batch)
            pred_summaries = self.model.generate(**batch)
            gt_summaries = batch['summary_text']
            for unique_id, query_text, pred_summary, gt_summary in zip(unique_ids, query_texts, pred_summaries, gt_summaries):
                writeable = f"Query:\n{'-'*100}\n{query_text}\n\n\nGround Truth:\n{'-'*100}\n{gt_summary}\n\n\nPred Summary:\n{'-'*100}\n{pred_summary}"
                with open(f"test-output-dir/{self.run_name}/{unique_id}.txt", 'w') as file:
                    file.write(writeable)
            
            pbar.update(1)
            
        pbar.close()
                
##=========WandB Setup=========##
wandb.login()
os.environ['WANDB_PROJECT'] = 'diff-sent-gen'
os.environ['WANDB_WATCH'] = 'all'
run_name = generate_slug(3)

##=========Loading Model=========##
def load_model(model_type, model_args, saved_dir):
    if model_type == 'naive-baseline': model =  NaiveBaseline(model_args['hf-backbone-name'])
    elif model_type == 'diffusion': model = DiffusionModel(model_args['beta-schedule'], model_args['diffusion-steps'], model_args['transformer-kwargs'], model_args['training-kwargs'], model_args['dtype'])
    else: raise NotImplementedError(f'Provide model-type {model_type} not implemented yet!')
    
    if saved_dir is not None:
        print(f"Loading model from {saved_dir}")
        state_dict = torch.load(f'{saved_dir}/pytorch_model.bin')
        model.load_state_dict(state_dict)
        
    return model
        

def load_collator_fn(model_type):
    if model_type == 'naive-baseline': return data_collator_for_naive_baseline
    elif model_type == 'diffusion': return data_collator_for_diffusum_baseline 
    else: raise NotImplementedError(f'Provide model-type {model_type} not implemented yet!')

if __name__ == '__main__':
    configuration = Configuration(run_name)
    model = load_model(configuration.MODEL_TYPE, configuration.load_model_args(), configuration.SAVED_DIR)
    
    if configuration.MODEL_PRETRAINED_PATH is not None:
        model.load_state_dict(torch.load(configuration.MODEL_PRETRAINED_PATH))
    
    train_dataset = NQForQfSDatasetV2(configuration.DATA_PATH, split='train', max_length=model.get_max_length())
    valid_dataset = NQForQfSDatasetV2(configuration.DATA_PATH, split='test', max_length=model.get_max_length())
    test_dataset = NQForQfSDatasetV2(configuration.DATA_PATH, split='test', max_length=model.get_max_length())
    
    training_args = TrainingArguments(
        output_dir=configuration.OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=configuration.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=configuration.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=configuration.GRAD_ACC,
        num_train_epochs=configuration.EPOCHS,
        learning_rate=configuration.LEARNING_RATE,
        bf16=configuration.USE_BF16,
        fp16=configuration.USE_FP16,
        evaluation_strategy="steps",
        eval_steps=configuration.EVAL_STEPS,
        save_strategy="steps",
        save_steps=configuration.EVAL_STEPS,
        dataloader_num_workers=8,
        log_level="error",
        logging_strategy="steps",
        logging_steps=configuration.LOG_STEPS, 
        lr_scheduler_type=configuration.LR_SCHEDULER,
        warmup_steps=configuration.LR_WARMUP,
        # optim=configuration.OPTIMIZER_NAME,
        run_name=configuration.RUN_NAME,
        weight_decay=configuration.WEIGHT_DECAY,
        max_grad_norm=configuration.MAX_GRAD_NORM,
        report_to='wandb'
    )
    
    collator_fn = load_collator_fn(configuration.MODEL_TYPE)
    
    optim = torch.optim.Adam(model.parameters(), lr=configuration.LEARNING_RATE)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers=(optim, None)
    )
    
    callback = CustomTrainerCallback()
    trainer.add_callback(callback)
    trainer.add_run_name(run_name)
    trainer.add_test_dataset(valid_dataset)
    
    # trainer.run_on_test_dataset()
    summary = trainer.train()
    trainer.save_model()
    trainer.run_on_test_dataset()
