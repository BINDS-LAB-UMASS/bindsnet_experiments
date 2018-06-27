import os
import sys
import scp
import yaml
import argparse
import pandas as pd

from paramiko import SSHClient
from scp import SCPClient


def main(model='diehl_and_cook_2015',
         data='mnist',
         param_string=None):
    """
    Downloads parameters for a particular network from the CICS swarm2 cluster.
    """
    f = os.path.join('..', 'credentials.yml')

    try:
        creds = yaml.load(open(f, 'r'))
    except FileNotFoundError:
        print('Create "credentials.yml" in top-level folder with username, password attributes.')
        sys.exit()

    username = creds['username']
    password = creds['password']

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('swarm2.cs.umass.edu', username=username, password=password)

    sftp = ssh.open_sftp()
    sftp.chdir(f'/mnt/nfs/work1/rkozma/{username}/experiments/params/{model}_{data}/')
    
    localpath = os.path.join('..', 'params', f'{model}_{data}')
    if not os.path.isdir(localpath):
        os.makedirs(localpath, exist_ok=True)

    sftp.get(param_string + '.p', os.path.join(localpath, param_string + '.p'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--param_string', type=str, required=True)
    args = parser.parse_args()

    model = args.model
    data = args.data
    param_string = args.param_string

    main(model, data, param_string)