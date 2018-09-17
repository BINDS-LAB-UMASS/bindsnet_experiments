import os
import sys
import scp
import yaml
import argparse
import pandas as pd

from paramiko import SSHClient
from scp import SCPClient


def main(model='diehl_and_cook_2015', data='mnist', cluster='swarm2', train=True):
    # language=rst
    """
    Downloads results CSV file from the CICS swarm2 cluster.
    """
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
    ssh.connect(f'{cluster}.cs.umass.edu', username=username, password=password)

    sftp = ssh.open_sftp()
    print(f'/mnt/nfs/work1/rkozma/{username}/experiments/results/{data}/{model}/')
    sftp.chdir(f'/mnt/nfs/work1/rkozma/{username}/experiments/results/{data}/{model}/')
    
    localpath = os.path.join('..', 'results', data, model)
    if not os.path.isdir(localpath):
        os.makedirs(localpath, exist_ok=True)

    sftp.get(f, os.path.join(localpath, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--cluster', type=str, default='swarm2')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    model = args.model
    data = args.data
    cluster = args.cluster
    train = args.train

    main(model, data, cluster, train)
