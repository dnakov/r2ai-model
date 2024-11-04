# ec2_training.py

import os
import argparse
import boto3
import base64
from datetime import datetime, timedelta
import time
import signal
import sys
import json
import pathlib
from botocore.exceptions import ClientError
import requests

INSTANCE_TYPE = 'g5.12xlarge' # 4xA10G 24GB mem per GPU ~2/hr spot
# INSTANCE_TYPE = 'g5.48xlarge' # 8xA10G 24GB mem per GPU ~$3/hr spot
# INSTANCE_TYPE = 'p4d.24xlarge' # 8xA100 40GB mem per GPU ~$10/hr spot
# INSTANCE_TYPE = 'p5.48xlarge' # 8xH100 80GB mem per GPU ~$50/hr spot
KEY_PAIR_NAME = 'kali'
# BUCKET_NAME = 'tc-radare2-training-data'
# AMI_BASE = 'ami-06b21ccaeff8cd686'
AMIS = {
    'us-east-1': 'ami-0aada1758622f91bb',
    'us-east-2': 'ami-0b9d4285990d49627',
    'us-west-2': 'ami-08e5fad56cda20dac',
}

ON_DEMAND_PRICES = {
    'g5.12xlarge': 4.00,  # Approximate on-demand prices
    'g5.48xlarge': 16.00,
    'p4d.24xlarge': 32.00,
    'p5.48xlarge': 98.00
}

def upload_to_s3(files, bucket_name=None):
    """Upload training files to S3 and return the bucket and paths"""
    s3 = boto3.client('s3')
        
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Using existing bucket: {bucket_name}")
    except s3.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            try:
                s3.create_bucket(
                    Bucket=bucket_name,
                )
                print(f"Created new bucket: {bucket_name}")
                
                # Add bucket versioning
                s3.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
            except Exception as e:
                print(f"Error creating bucket: {str(e)}")
                raise
        else:
            print(f"Error checking bucket: {str(e)}")
            raise
    # Upload files
    s3_paths = {}
    for file_path in files:
        key = os.path.basename(file_path)
        try:
            s3.upload_file(file_path, bucket_name, key)
            s3_paths[key] = f"s3://{bucket_name}/{key}"
        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")
            raise

    return bucket_name, s3_paths

def create_efs_filesystem(ec2_client, efs_client, vpc_id, security_group_id):
    """Create EFS filesystem and mount targets"""
    try:
        # Check if EFS filesystem already exists
        existing_filesystems = efs_client.describe_file_systems()['FileSystems']
        existing_fs = next((fs for fs in existing_filesystems if fs['Tags'][0]['Value'] == 'TrainingCheckpoints'), None)
        
        if existing_fs:
            print(f"Using existing EFS filesystem: {existing_fs['FileSystemId']}")
            fs_id = existing_fs['FileSystemId']
        else:
            # Create new EFS filesystem
            response = efs_client.create_file_system(
                PerformanceMode='generalPurpose',
                ThroughputMode='bursting',
                Tags=[{'Key': 'Name', 'Value': 'TrainingCheckpoints'}]
            )
            fs_id = response['FileSystemId']
            print(f"Created new EFS filesystem: {fs_id}")
        
            # Wait for filesystem to become available
            while True:
                response = efs_client.describe_file_systems(FileSystemId=fs_id)
                if response['FileSystems'][0]['LifeCycleState'] == 'available':
                    break
                time.sleep(5)
        
        # Get subnet IDs from VPC
        subnets = ec2_client.describe_subnets(
            Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
        )['Subnets']
        
        # Create mount targets in each subnet if they don't exist
        existing_mount_targets = efs_client.describe_mount_targets(FileSystemId=fs_id)['MountTargets']
        for subnet in subnets:
            if not any(mt['SubnetId'] == subnet['SubnetId'] for mt in existing_mount_targets):
                efs_client.create_mount_target(
                    FileSystemId=fs_id,
                    SubnetId=subnet['SubnetId'],
                    SecurityGroups=[security_group_id]
                )
                print(f"Created mount target in subnet: {subnet['SubnetId']}")
            else:
                print(f"Mount target already exists in subnet: {subnet['SubnetId']}")
        
        return fs_id
    except Exception as e:
        print(f"Error creating or using EFS filesystem: {str(e)}")
        raise

def update_security_group_for_efs(ec2_client, security_group_id):
    """Add inbound rule for EFS if it doesn't exist"""
    try:
        security_group = ec2_client.describe_security_groups(GroupIds=[security_group_id])['SecurityGroups'][0]
        efs_rule_exists = any(
            rule['FromPort'] == 2049 and rule['ToPort'] == 2049 and rule['IpProtocol'] == 'tcp'
            for rule in security_group['IpPermissions']
        )
        
        if not efs_rule_exists:
            ec2_client.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[{
                    'FromPort': 2049,
                    'ToPort': 2049,
                    'IpProtocol': 'tcp',
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )
            print("Added EFS inbound rule to security group")
        else:
            print("EFS inbound rule already exists in security group")
    except ClientError as e:
        print(f"Error updating security group: {str(e)}")
        raise

def get_user_data_script(s3_paths, args, efs_id):
    """Generate user data script for EC2 instance initialization"""
    jupyter_sh = open('jupyter.sh', 'r').read()
    return """#!/bin/bash

mkdir -p /app
cd /app
apt update
apt install -y stunnel4
aws s3 cp s3://{bucket}/amazon-efs-utils-x64.deb ./
dpkg -i amazon-efs-utils-x64.deb
mkdir -p /mnt/efs
mount -t efs {efs_id} /mnt/efs
mkdir -p /mnt/efs/checkpoints
mkdir -p /mnt/efs/output/metrics
mkdir -p /mnt/efs/model-{ts}
mkdir -p /mnt/efs/model_cache
chown -R ubuntu:ubuntu /mnt/efs

export CHECKPOINT_DIR=/mnt/efs/checkpoints
export METRICS_DIR=/mnt/efs/output/metrics
export HF_HUB_CACHE="/mnt/efs/model_cache"

source activate pytorch
{jupyter_sh}
pip install setuptools==70.3.0
pip install -U accelerate torch trl transformers tensorboard pandas datasets
chown -R ubuntu:ubuntu /home/ubuntu

""".format(
    ts=datetime.now().strftime("%Y%m%d-%H%M%S"),
    efs_id=efs_id,
    bucket=args.bucket,
    wandb_api_key=os.getenv('WANDB_API_KEY'),
    hf_token=os.getenv('HF_TOKEN'),
    jupyter_sh=jupyter_sh
)

def verify_training_files(files):
    """Verify all required training files exist"""
    missing_files = []
    for file_path in files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required training files: {', '.join(missing_files)}"
        )

class EC2SpotManager:
    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.instance_id = None
        self.spot_request_id = None
        self.state_file = pathlib.Path('ec2_training_state2.json')
        
        # Try to load existing state
        self.load_state()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.cleanup_handler)
        signal.signal(signal.SIGTERM, self.cleanup_handler)

    def save_state(self):
        """Save current state to file"""
        state = {
            'instance_id': self.instance_id,
            'spot_request_id': self.spot_request_id,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def load_state(self):
        """Load previous state if exists"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.instance_id = state.get('instance_id')
                self.spot_request_id = state.get('spot_request_id')
                print(f"Loaded previous state: instance_id={self.instance_id}, "
                      f"spot_request_id={self.spot_request_id}")
                return True
            except Exception as e:
                print(f"Error loading state: {str(e)}")
        return False

    def clear_state(self):
        """Clear the state file"""
        if self.state_file.exists():
            self.state_file.unlink()

    def cleanup_handler(self, signum, frame):
        print("\nReceived termination signal. Cleaning up resources...")
        self.cleanup_resources()
        self.clear_state()
        sys.exit(0)

    def cleanup_resources(self):
        """Cleanup EC2 resources including EBS volumes"""
        try:
            if self.instance_id:
                # Get volume information before terminating the instance
                response = self.ec2.describe_instances(InstanceIds=[self.instance_id])
                volumes = response['Reservations'][0]['Instances'][0].get('BlockDeviceMappings', [])
                volume_ids = [v['Ebs']['VolumeId'] for v in volumes if 'Ebs' in v]
                
                print(f"Terminating instance {self.instance_id}")
                self.ec2.terminate_instances(InstanceIds=[self.instance_id])
                
                # Wait for instance to terminate
                waiter = self.ec2.get_waiter('instance_terminated')
                waiter.wait(InstanceIds=[self.instance_id])
                
                # Delete associated EBS volumes
                for volume_id in volume_ids:
                    try:
                        print(f"Deleting EBS volume {volume_id}")
                        self.ec2.delete_volume(VolumeId=volume_id)
                    except Exception as e:
                        print(f"Error deleting volume {volume_id}: {str(e)}")
            
            if self.spot_request_id:
                print(f"Cancelling spot request {self.spot_request_id}")
                self.ec2.cancel_spot_instance_requests(
                    SpotInstanceRequestIds=[self.spot_request_id]
                )
            self.clear_state()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def verify_active_resources(self):
        """Verify if previously saved resources are still active"""
        try:
            if self.instance_id:
                response = self.ec2.describe_instances(InstanceIds=[self.instance_id])
                state = response['Reservations'][0]['Instances'][0]['State']['Name']
                if state not in ['pending', 'running']:
                    print(f"Instance {self.instance_id} is in state {state}")
                    self.instance_id = None
                    
            if self.spot_request_id:
                response = self.ec2.describe_spot_instance_requests(
                    SpotInstanceRequestIds=[self.spot_request_id]
                )
                state = response['SpotInstanceRequests'][0]['State']
                if state not in ['pending-evaluation', 'pending-fulfillment', 'fulfilled']:
                    print(f"Spot request {self.spot_request_id} is in state {state}")
                    self.spot_request_id = None
                    
            if not self.instance_id and not self.spot_request_id:
                self.clear_state()
                return False
                
            return bool(self.instance_id or self.spot_request_id)
            
        except Exception as e:
            print(f"Error verifying resources: {str(e)}")
            self.clear_state()
            return False
        
    def create_iam_role_and_profile(self, bucket_name):
        """Create IAM role and instance profile for EC2 if they don't exist"""
        iam = boto3.client('iam')
        role_name = 'EC2_S3_Access_Role'
        profile_name = 'EC2_S3_Access_Profile'

        try:
            # Check if role already exists
            iam.get_role(RoleName=role_name)
            print(f"IAM role {role_name} already exists")
        except iam.exceptions.NoSuchEntityException:
            # Create role
            trust_policy = {
              "Version": "2012-10-17",
              "Statement": [{
                  "Effect": "Allow",
                  "Principal": {"Service": "ec2.amazonaws.com"},
                  "Action": "sts:AssumeRole"
              }]
            }
          
            print(f"Creating IAM role {role_name}")
            iam.create_role(
              RoleName=role_name,
              AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
          
            # Attach S3 policy
            # Create policy document for specific bucket access
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*"
                    ]
                }]
            }
            
            # Create and attach the bucket-specific policy
            policy_name = f"{role_name}_bucket_policy"
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(bucket_policy)
            )

        try:
            # Check if instance profile exists
            iam.get_instance_profile(InstanceProfileName=profile_name)
            print(f"Instance profile {profile_name} already exists")
        except iam.exceptions.NoSuchEntityException:
            # Create instance profile and add role
            print(f"Creating instance profile {profile_name}")
            iam.create_instance_profile(InstanceProfileName=profile_name)
            
            try:
                iam.add_role_to_instance_profile(
                    InstanceProfileName=profile_name,
                    RoleName=role_name
              )
            except iam.exceptions.LimitExceededException:
              print("Role already added to instance profile")

        # Wait for the instance profile to be ready
        time.sleep(10)
        return profile_name

    def setup_ec2_training(self, args):
        """Configure and launch EC2 instance for training"""
        # Check if we have existing resources
        if self.load_state() and self.verify_active_resources():
            print("Resuming monitoring of existing resources...")
            self.monitor_instance()
            return self.instance_id

        # List of files needed for training
        training_files = [
            'ec2/amazon-efs-utils-x64.deb',
            args.data_path
        ]

        # Verify training files
        verify_training_files(training_files)

        # Upload files to S3
        bucket_name, s3_paths = upload_to_s3(training_files, args.bucket)
        args.bucket = bucket_name  # Store bucket name for user data script
        efs_client = boto3.client('efs')
        vpc_id = self.ec2.describe_vpcs()['Vpcs'][0]['VpcId']
        security_group_id = None

        def create_ingress_rules(security_group, ip):
            security_group_id = security_group['GroupId']
            self.ec2.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': f'{my_ip}/32'}]
                    }
                ]
            )
            # Add inbound rule for Jupyter access
            self.ec2.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 8888,
                        'ToPort': 8888,
                        'IpRanges': [{'CidrIp': f'{my_ip}/32'}]
                    }
                ]
            )
            print(f"Created ingress rules for {ip}")


        def delete_ingress_rules(security_group, ip):
            security_group_id = security_group['GroupId']
            existing_rules = security_group['IpPermissions']
            for rule in existing_rules:
                if (rule['FromPort'] == 22 and rule['ToPort'] == 22) or (rule['FromPort'] == 8888 and rule['ToPort'] == 8888):
                    self.ec2.revoke_security_group_ingress(
                        GroupId=security_group_id,
                        IpPermissions=[rule]
                    )
            print(f"Deleted ingress rules for {ip}")
        my_ip = requests.get('https://api.ipify.org').text
        try:
            response = self.ec2.create_security_group(
                GroupName='LLMTrainingSecurityGroup',
                Description='Security group for LLM training on EC2'
            )
            security_group_id = response['GroupId']
            create_ingress_rules(security_group_id, my_ip)
            print(f"Created security group: {security_group_id}")
        except self.ec2.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidGroup.Duplicate':
                
                print("Security group already exists. Using existing group.")

                security_group = self.ec2.describe_security_groups(
                    GroupNames=['LLMTrainingSecurityGroup']
                )['SecurityGroups'][0]
                security_group_id = security_group['GroupId']
                delete_ingress_rules(security_group, my_ip)
                create_ingress_rules(security_group, my_ip)
            else:
                raise e
        efs_id = create_efs_filesystem(self.ec2, efs_client, vpc_id, security_group_id)
        update_security_group_for_efs(self.ec2, security_group_id)            
        user_data = get_user_data_script(s3_paths, args, efs_id)
        iam_profile = self.create_iam_role_and_profile(bucket_name)

        # Update launch specification
        launch_specification = {
            'ImageId': AMIS[os.environ['AWS_DEFAULT_REGION']],
            # 'ImageId': AMI_BASE,
            'InstanceType': INSTANCE_TYPE,
            'KeyName': KEY_PAIR_NAME,
            'SecurityGroupIds': [security_group_id],
            'IamInstanceProfile': {
                'Name': iam_profile
            },
            'BlockDeviceMappings': [{
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 100,
                    'VolumeType': 'gp3',
                    'DeleteOnTermination': True
                }
            }],
            'UserData': base64.b64encode(user_data.encode()).decode()
        }
        
        try:
            if args.on_demand:
                # Launch on-demand instance
                response = self.ec2.run_instances(
                    ImageId=launch_specification['ImageId'],
                    InstanceType=launch_specification['InstanceType'],
                    KeyName=launch_specification['KeyName'],
                    SecurityGroupIds=launch_specification['SecurityGroupIds'],
                    IamInstanceProfile=launch_specification['IamInstanceProfile'],
                    BlockDeviceMappings=launch_specification['BlockDeviceMappings'],
                    UserData=user_data,
                    MinCount=1,
                    MaxCount=1
                )
                
                self.instance_id = response['Instances'][0]['InstanceId']
                self.save_state()
                print(f"On-demand instance launched: {self.instance_id}")
                
                # Wait for instance to be running
                waiter = self.ec2.get_waiter('instance_running')
                waiter.wait(InstanceIds=[self.instance_id])

            else:
                # Existing spot instance code
                response = self.ec2.request_spot_instances(
                    InstanceCount=1,
                    LaunchSpecification=launch_specification,
                    ValidUntil=datetime.now() + timedelta(hours=24),
                    Type='one-time'
                )
                
                self.spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                self.save_state()
                print(f"Spot instance request created: {self.spot_request_id}")
                
                # Wait for spot instance to be fulfilled
                waiter = self.ec2.get_waiter('spot_instance_request_fulfilled')
                waiter.wait(
                    SpotInstanceRequestIds=[self.spot_request_id],
                    WaiterConfig={'Delay': 20, 'MaxAttempts': 30}
                )
                
                spot_request = self.ec2.describe_spot_instance_requests(
                    SpotInstanceRequestIds=[self.spot_request_id]
                )['SpotInstanceRequests'][0]
                
                self.instance_id = spot_request['InstanceId']
                self.save_state()
                print(f"Spot instance launched: {self.instance_id}")

            # Get instance public IP address
            instance_info = self.ec2.describe_instances(InstanceIds=[self.instance_id])
            public_ip = instance_info['Reservations'][0]['Instances'][0]['PublicIpAddress']
            print(f"\nInstance public IP: {public_ip}")
            print(f"Jupyter URL: http://{public_ip}:8888 (might take a few min)")

            # Show pricing information
            instance_type = launch_specification['InstanceType']
            if args.on_demand:
                print(f"On-demand price: ~${ON_DEMAND_PRICES[instance_type]}/hour")
            else:
                print(f"Spot price: ~${ON_DEMAND_PRICES[instance_type]/2}/hour (estimated)")

            self.ec2.create_tags(
                Resources=[self.instance_id],
                Tags=[
                    {'Key': 'Name', 'Value': f'training-{args.wandb_project}'},
                    {'Key': 'Project', 'Value': args.wandb_project}
                ]
            )

            # Monitor instance status until training completes or fails
            self.monitor_instance()
            
            return self.instance_id
            
        except Exception as e:
            print(f"Error launching instance: {str(e)}")
            self.cleanup_resources()
            raise

    def monitor_instance(self):
        """Monitor instance status and cleanup when training completes or fails"""
        try:
            while True:
                response = self.ec2.describe_instances(InstanceIds=[self.instance_id])
                state = response['Reservations'][0]['Instances'][0]['State']['Name']
                
                if state == 'terminated' or state == 'stopped' or state == 'shutting-down':
                    print(f"Instance {self.instance_id} has {state}. Cleaning up...")
                    self.cleanup_resources()
                    break
                elif state == 'running':
                    # You could add additional monitoring here
                    # For example, checking training logs or metrics
                    pass
                
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            print(f"Error monitoring instance: {str(e)}")
            self.cleanup_resources()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to JSONL data file', 
                       default='../data/radare2/radare2_train.jsonl')
    parser.add_argument('--model_name', default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--wandb_project', default='radare2-llama3.2-1b')
    parser.add_argument('--bucket', help='S3 bucket for training files', 
                       default='tc-radare2-training-data')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up any existing resources and exit')
    parser.add_argument('--on_demand', action='store_true',
                       help='Use on-demand instance instead of spot')
    
    args = parser.parse_args()
    
    manager = EC2SpotManager()
    
    if args.cleanup:
        print("Cleaning up resources...")
        manager.cleanup_resources()
        sys.exit(0)
    
    instance_id = manager.setup_ec2_training(args)

if __name__ == "__main__":
    main()