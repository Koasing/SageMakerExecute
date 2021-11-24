import argparse
import json
import logging
import os
import sys

from pprint import pprint

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms

from Model import SimpleModel

logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # data preparation
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # forward pass
        output = model(data)

        # loss calculation
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # training log
        if batch_idx % args.log_interval == 0:
            logger.info(f'Train Epoch: {epoch} '
                        f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                        f'Loss: {loss.item():.6f}')
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # data preparation
            data, target = data.to(device), target.to(device)

            # forward pass
            output = model(data)

            # loss calculation
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            # performance calculation
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(f'Test set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(test_loader.dataset)} '
                f'({100. * correct / len(test_loader.dataset):.0f}%)')


def main(args):
    logger.info('Arguments -----')
    for key, value in vars(args).items():
        logger.info(f'{key} = {value}')

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST(args.mnist_dir, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(args.mnist_dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = SimpleModel().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

        if epoch % args.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            },
                f'{args.artifacts_dir}/model_{epoch}.pth')

    # save final model
    torch.save(model.state_dict(), f'{args.model_dir}/mnist_cnn.pt')


if __name__ == '__main__':
    from sm_env_wrapper import set_sm_environ
    set_sm_environ()

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint-interval', type=int, default=1, metavar='N', help='how many epochs to wait before saving model checkpoint')
    parser.add_argument('--backend', type=str, default=None, help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment.
    # For some useful list, see https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-cpus', type=int, default=os.environ['SM_NUM_CPUS'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])
    # parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--mnist-dir', type=str, default=os.environ['SM_CHANNEL_MNIST'])

    # save final model file in this directory.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # save any artifacts in this directory.
    parser.add_argument('--artifacts-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    # checkpoint directory is not included in the Environment Variables. Use default.
    # save any intermediate checkpoint archives in this directory.
    parser.add_argument('--checkpoints-dir', type=str, default='/opt/ml/checkpoints/')

    args = parser.parse_args()
    main(args)
