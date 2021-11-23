import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


sagemaker_session = sagemaker.Session(default_bucket='sagemaker-your_id')

# IAM Role requires SageMakerFullAccess & S3FullAccess permissions.
role = 'arn:aws:iam::your_id:role/SageMakerExecution'

bucket = sagemaker_session.default_bucket()
dataset_prefix = 'dataset/dogface_v3'
checkpoints_prefix = f'checkpoints'
output_prefix = f'trained_models'

hyper_params = {
        'epochs': 1,
        'lr': 0.01,
}

# Available framework & python version tables:
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md#general-framework-containers-ec2-ecs-eks--sm-support

job_name='SimpleTrain'

estimator = PyTorch(
        base_job_name=job_name,
        entry_point='train.py',
        source_dir='src',
        hyperparameters=hyper_params,
        framework_version='1.9.0',
        py_version='py38',
        #image_uri='763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.9.1-gpu-py38-cu111-ubuntu20.04',
        output_path=f's3://{bucket}/{output_prefix}',
        checkpoint_s3_uri=f's3://{bucket}/{checkpoints_prefix}',
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        sagemaker_session=sagemaker_session)

estimator.fit(inputs={'training': f's3://{bucket}/{dataset_prefix}/training',
                      'test': f's3://{bucket}/{dataset_prefix}/test',
                      'validation': f's3://{bucket}/{dataset_prefix}/validation'})
