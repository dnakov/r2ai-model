# main.py

import os
import argparse
import sagemaker
from sagemaker.huggingface import HuggingFace
from datetime import datetime

def setup_sagemaker_training(args):
    """Configure and launch SageMaker training job"""
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    # Upload training data to S3
    train_data_s3 = sagemaker_session.upload_data(
        args.data_path,
        bucket=sagemaker_session.default_bucket(),
        key_prefix='radare2-training-data'
    )
    
    checkpoint_s3_uri = f's3://{sagemaker_session.default_bucket()}/checkpoints'
    
    # Define metrics for tracking
    metric_definitions = [
        {'Name': 'train:loss', 'Regex': "'loss': ([0-9\\.]+)"},
        {'Name': 'eval:loss', 'Regex': "'eval_loss': ([0-9\\.]+)"},
        {'Name': 'checkpoint_step', 'Regex': "'checkpoint_saved_at_step': ([0-9]+)"},
    ]
    
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./',
        instance_type='ml.g5.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.36',
        pytorch_version='2.1',
        py_version='py310',
        base_job_name=f'radare2-llama3-2-1b-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        metric_definitions=metric_definitions,
        hyperparameters={
            'model_name': args.model_name,
            'epochs': args.epochs,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'wandb_project': args.wandb_project,
            'lora_r': 32,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'warmup_ratio': 0.05,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 0.3,
            'checkpoint_frequency': 100,
        },
        environment={
            'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
            'CHECKPOINT_DIR': '/opt/ml/checkpoints',
            'METRICS_DIR': '/opt/ml/output/metrics',
            'HF_TOKEN': os.getenv('HF_TOKEN'),
        },
        use_spot_instances=True,
        max_wait=24 * 60 * 60,
        max_run=23 * 60 * 60,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path='/opt/ml/checkpoints'
    )
    
    huggingface_estimator.fit({'training': train_data_s3}, wait=True)
    return huggingface_estimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to JSONL data file', default='../data/radare2/radare2_train.jsonl')
    parser.add_argument('--model_name', default='meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--wandb_project', default='radare2-llama3.2-1b')
    
    args = parser.parse_args()
    setup_sagemaker_training(args)

if __name__ == "__main__":
    main()