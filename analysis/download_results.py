import os
import sys
import scp
import yaml
import argparse
import pandas as pd

from paramiko import SSHClient
from scp import SCPClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    model = args.model
    data = args.data
    train = args.train

    f = os.path.join('..', 'credentials.yml')

    try:
        creds = yaml.load(open(f, 'r'))
    except FileNotFoundError:
        print('Create "credentials.yml" in top-level folder with username, password attributes.')
        sys.exit()

    username = creds['username']
    password = creds['password']

    if train:
        f = 'train.csv'
    else:
        f = 'test.csv'

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('swarm2.cs.umass.edu', username=username, password=password)

    sftp = ssh.open_sftp()
    sftp.chdir(f'/mnt/nfs/work1/rkozma/{username}/experiments/results/{model}_{data}/')
    
    localpath = os.path.join('..', 'results', f'{model}_{data}')
    if not os.path.isdir(localpath):
        os.makedirs(localpath, exist_ok=True)

    sftp.get(f, os.path.join(localpath, f))
