import os
import json

import multiprocessing as mp

# Container environment variables.
# For some useful list, see https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md


SM_ENV_INIT = {
    'SM_HOSTS': '["localhost"]',
    'SM_CURRENT_HOST': 'localhost',
    'SM_NUM_CPUS': str(mp.cpu_count()),
    'SM_NUM_GPUS': '0',
    'SM_CHANNELS': '["mnist"]',
    'SM_CHANNEL_MNIST': './dataset/MNIST',
    'SM_MODEL_DIR': './model',
    'SM_OUTPUT_DATA_DIR': './output'
}


def set_sm_environ():
    for key, value in SM_ENV_INIT.items():
        os.environ.setdefault(key, value)
