# hpo.py

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
from sagemaker.tuner import (
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes
)

class HPOManager:
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
        self.sm_client = boto3.client('sagemaker')
        
        # Create output directories
        self.output_dir = Path('hpo_results')
        self.output_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hpo.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('HPOManager')

    def get_hyperparameter_ranges(self):
        """Define hyperparameter ranges for tuning"""
        return {
            # LoRA parameters
            'lora_r': IntegerParameter(16, 64, scaling_type='linear'),
            'lora_alpha': IntegerParameter(16, 64, scaling_type='linear'),
            'lora_dropout': ContinuousParameter(0.0, 0.2, scaling_type='linear'),
            
            # Training parameters
            'learning_rate': ContinuousParameter(1e-5, 5e-4, scaling_type='logarithmic'),
            'batch_size': IntegerParameter(8, 32, scaling_type='linear'),
            'warmup_ratio': ContinuousParameter(0.01, 0.1, scaling_type='linear'),
            'weight_decay': ContinuousParameter(0.01, 0.1, scaling_type='linear'),
            
            # Architecture choices
            'gradient_accumulation_steps': IntegerParameter(1, 4, scaling_type='linear'),
            
            # Target modules combinations
            'target_modules': CategoricalParameter([
                'q_proj,k_proj,v_proj,o_proj',  # Attention only
                'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',  # Include MLP
                'q_proj,k_proj,v_proj,o_proj,gate_proj',  # Partial MLP
            ])
        }

    def create_estimator(self):
        """Create SageMaker estimator for training"""
        
        # Define metrics to track
        metric_definitions = [
            {'Name': 'train:loss', 'Regex': "'loss': ([0-9\\.]+)"},
            {'Name': 'eval:loss', 'Regex': "'eval_loss': ([0-9\\.]+)"},
            {'Name': 'gpu:memory', 'Regex': "'gpu_memory_allocated': ([0-9\\.]+)"},
            {'Name': 'learning:rate', 'Regex': "'learning_rate': ([0-9\\.]+)"}
        ]
        
        return HuggingFace(
            entry_point='train.py',
            source_dir='./scripts',
            instance_type=self.args.instance_type,
            instance_count=1,
            role=self.role,
            transformers_version='4.28.1',
            pytorch_version='2.0.0',
            py_version='py39',
            base_job_name=f'llama-hpo-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            metric_definitions=metric_definitions,
            hyperparameters={
                'model_name': self.args.model_name,
                'epochs': self.args.epochs,
                'wandb_project': self.args.wandb_project,
                'max_grad_norm': 0.3,
            },
            environment={
                'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
                'CHECKPOINT_DIR': '/opt/ml/checkpoints',
                'METRICS_DIR': '/opt/ml/output/metrics',
            },
            use_spot_instances=True,
            max_wait=7200,  # 2 hours
            max_run=43200,  # 12 hours
            checkpoint_s3_uri=f's3://{self.session.default_bucket()}/checkpoints',
            checkpoint_local_path='/opt/ml/checkpoints'
        )

    def setup_tuner(self, estimator):
        """Configure hyperparameter tuner"""
        
        # Setup warm start if previous tuning job exists
        warm_start_config = None
        if self.args.warm_start_job:
            warm_start_config = WarmStartConfig(
                warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
                parents=[self.args.warm_start_job]
            )
        
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='eval:loss',
            hyperparameter_ranges=self.get_hyperparameter_ranges(),
            max_jobs=self.args.max_jobs,
            max_parallel_jobs=self.args.max_parallel_jobs,
            objective_type='Minimize',
            strategy='Bayesian',
            early_stopping_type='Auto',
            warm_start_config=warm_start_config
        )
        
        return tuner

    def analyze_results(self, tuning_job_name):
        """Analyze HPO results"""
        response = self.sm_client.list_training_jobs_for_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )
        
        # Collect results
        results = []
        for job in response['TrainingJobSummaries']:
            job_name = job['TrainingJobName']
            job_detail = self.sm_client.describe_training_job(TrainingJobName=job_name)
            
            results.append({
                'job_name': job_name,
                'hyperparameters': job_detail['HyperParameters'],
                'objective_value': job_detail['FinalMetricDataList'][0]['Value'],
                'status': job_detail['TrainingJobStatus'],
                'duration': str(job_detail['TrainingTimeInSeconds'] / 3600) + ' hours',
                'billable_time': str(job_detail['BillableTimeInSeconds'] / 3600) + ' hours',
                'cost_estimate': (job_detail['BillableTimeInSeconds'] / 3600) * 
                                self.get_instance_cost(job_detail['ResourceConfig']['InstanceType'])
            })
        
        # Sort by objective value
        results.sort(key=lambda x: x['objective_value'])
        
        # Save results
        output_file = self.output_dir / f'hpo_results_{tuning_job_name}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

    def get_instance_cost(self, instance_type):
        """Get approximate cost per hour for instance type"""
        # Add your instance costs here
        costs = {
            'ml.g5g.xlarge': 0.126,
            'ml.g5.xlarge': 0.302,
            'ml.p3.2xlarge': 0.918
        }
        return costs.get(instance_type, 1.0)  # Default to 1.0 if unknown

    def monitor_jobs(self, tuner):
        """Monitor HPO progress"""
        tuning_job_name = tuner.latest_tuning_job.name
        self.logger.info(f"Hyperparameter tuning job: {tuning_job_name}")
        
        while True:
            status = self.sm_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name
            )
            
            self.logger.info(
                f"Status: {status['HyperParameterTuningJobStatus']}, "
                f"Jobs: {status['TrainingJobStatusCounters']}"
            )
            
            if status['HyperParameterTuningJobStatus'] in ['Completed', 'Failed', 'Stopped']:
                break
                
            time.sleep(300)  # Check every 5 minutes
        
        # Analyze results
        results = self.analyze_results(tuning_job_name)
        
        # Print best result
        best_job = results[0]
        self.logger.info("\nBest configuration:")
        self.logger.info(f"Objective value: {best_job['objective_value']}")
        self.logger.info("Hyperparameters:")
        for name, value in best_job['hyperparameters'].items():
            self.logger.info(f"  {name}: {value}")
        
        return results

    def run(self):
        """Run hyperparameter optimization"""
        self.logger.info("Starting hyperparameter optimization")
        
        # Upload training data to S3
        train_data_s3 = self.session.upload_data(
            self.args.data_path,
            bucket=self.session.default_bucket(),
            key_prefix='llama-training-data'
        )
        
        # Create estimator
        estimator = self.create_estimator()
        
        # Setup tuner
        tuner = self.setup_tuner(estimator)
        
        # Start tuning
        tuner.fit({'training': train_data_s3}, wait=False)
        
        # Monitor progress
        results = self.monitor_jobs(tuner)
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to TSV data file')
    parser.add_argument('--model_name', default='meta-llama/Llama-1b')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--wandb_project', default='llama-hpo')
    parser.add_argument('--max_jobs', type=int, default=20,
                       help='Maximum number of training jobs to launch')
    parser.add_argument('--max_parallel_jobs', type=int, default=2,
                       help='Maximum number of parallel training jobs')
    parser.add_argument('--instance_type', default='ml.g5g.xlarge',
                       help='SageMaker instance type')
    parser.add_argument('--warm_start_job', type=str, default=None,
                       help='Previous HPO job name for warm start')
    
    args = parser.parse_args()
    
    hpo_manager = HPOManager(args)
    results = hpo_manager.run()
    
    # Print summary
    print("\nHPO Summary:")
    print(f"Total jobs: {len(results)}")
    print(f"Best loss: {results[0]['objective_value']}")
    print(f"Total cost estimate: ${sum(r['cost_estimate'] for r in results):.2f}")
    print(f"Results saved to: {hpo_manager.output_dir}")

if __name__ == "__main__":
    main()