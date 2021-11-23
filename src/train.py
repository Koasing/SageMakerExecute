import argparse
import json
import logging
import os
import sys

from pprint import pprint

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from torchvision import datasets, transforms

from Model.simple import SimpleModel

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('SM_LOG_LEVEL', logging.INFO))
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(args):
    logger.info('Start training...')

    logger.info('Arguments -----')
    for key, value in vars(args).items():
        logger.info(f'{key} = {value}')


if __name__ == '__main__':
    from sm_env_wrapper import set_sm_environ
    set_sm_environ()

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None, help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment.
    # For some useful list, see https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-cpus', type=int, default=os.environ['SM_NUM_CPUS'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    # save final model file in this directory.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # save any artifacts in this directory.
    parser.add_argument('--artifacts-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    # checkpoint directory is not included in the Environment Variables. Use default.
    # save any intermediate checkpoint archives in this directory.
    parser.add_argument('--checkpoints-dir', type=str, default='/opt/ml/checkpoints/')

    args = parser.parse_args()
    train(args)
