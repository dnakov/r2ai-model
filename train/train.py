# train.py

import os
import json
import torch
import wandb
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from datetime import datetime
import argparse

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)

from checkpoint_manager import CheckpointManager
from utils import setup_logging, set_seed, calculate_memory_usage

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_ratio: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    checkpoint_frequency: int
    target_modules: List[str] = None
    max_length: int = 512
    
    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            checkpoint_frequency=args.checkpoint_frequency,
            target_modules=args.target_modules.split(',') if args.target_modules else [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]
        )

class QLoRATrainer(Trainer):
    """Enhanced trainer with checkpoint management and spot instance handling"""
    def __init__(self, checkpoint_manager: CheckpointManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger('QLoRATrainer')
        
    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        # Try to resume from best checkpoint if not specified
        if resume_from_checkpoint is None:
            best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
            if best_checkpoint:
                self.logger.info(f"Resuming from best checkpoint: {best_checkpoint}")
                resume_from_checkpoint = best_checkpoint
        
        starting_step = 0
        if resume_from_checkpoint:
            starting_step = self.checkpoint_manager.load_checkpoint(
                self, resume_from_checkpoint
            ) or 0
            
        def spot_checkpoint_handler(args, state, control, model=None, **kwargs):
            """Handle checkpointing with validation and monitoring"""
            if state.global_step % self.args.save_steps == 0:
                # Get validation loss if available
                val_loss = None
                if state.log_history:
                    val_losses = [log.get('eval_loss') for log in state.log_history]
                    val_loss = next((l for l in val_losses if l is not None), None)
                
                # Monitor memory usage
                memory_usage = calculate_memory_usage()
                if memory_usage:
                    self.log({
                        'gpu_memory_allocated': memory_usage['allocated'],
                        'gpu_memory_reserved': memory_usage['reserved']
                    })
                
                # Determine if this should be a partial save
                is_partial = state.global_step % (self.args.save_steps * 5) != 0
                
                try:
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        self,
                        state.global_step,
                        self.args,
                        state.log_history[-1]['loss'] if state.log_history else None,
                        validation_loss=val_loss,
                        partial=is_partial
                    )
                    
                    self.log({
                        'checkpoint_step': state.global_step,
                        'checkpoint_path': str(checkpoint_path),
                        'checkpoint_type': 'partial' if is_partial else 'full',
                        'validation_loss': val_loss
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to save checkpoint: {str(e)}")
                
            return control
            
        self.add_callback(spot_checkpoint_handler)
        
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            *args,
            **kwargs
        )

def setup_model(config: TrainingConfig):
    """Initialize model with QLoRA configuration"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    return model

def prepare_dataset(data_path: str, tokenizer, config: TrainingConfig):
    """Prepare dataset from JSONL file"""
    df = pd.read_json(data_path, lines=True)
    
    def format_prompt(row):
      text = "<|begin_of_text|>"
      for message in row['messages']:
        text += f"<|start_header_id|>{message['role']}<|end_header_id|>{message['content']}<|eot_id|>"
      return text
    
    df['text'] = df.apply(format_prompt, axis=1)
    dataset = Dataset.from_pandas(df[['text']])
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.max_length
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split for validation
    split = tokenized_dataset.train_test_split(test_size=0.1)
    return split['train'], split['test']

def train(args):
    """Main training function"""
    logger = setup_logging(
        Path(args.output_dir) / 'logs',
        'training'
    )
    
    # Set random seed
    set_seed(42)
    
    # Initialize wandb
    if args.wandb_project:
        wandb.init(project=args.wandb_project)
    
    # Load config
    config = TrainingConfig.from_args(args)
    logger.info(f"Training config: {asdict(config)}")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="right",
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(
        args.train,
        tokenizer,
        config
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.environ['CHECKPOINT_DIR'],
        metrics_dir=os.environ['METRICS_DIR'],
        max_checkpoints=5,
        validation_frequency=5,
        merge_threshold=3,
        min_improvement=0.01
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=10,
        save_steps=config.checkpoint_frequency,
        eval_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="wandb" if args.wandb_project else None,
        save_total_limit=3,
        remove_unused_columns=False,
        optim="paged_adamw_32bit"
    )
    
    # Initialize model
    model = setup_model(config)
    logger.info(f"Model setup complete. Device map: {model.hf_device_map}")
    
    # Initialize trainer
    trainer = QLoRATrainer(
        checkpoint_manager=checkpoint_manager,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
        ]
    )
    
    try:
        # Start training
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Save final model
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Log final metrics
        if args.wandb_project:
            wandb.finish()
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--checkpoint_frequency', type=int, default=100)
    parser.add_argument('--target_modules', type=str, 
                       default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--huggingface_token', type=str, default=os.environ.get('HF_TOKEN'))
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)