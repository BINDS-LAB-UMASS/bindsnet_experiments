import os
import sys
import yaml
import argparse

from tqdm import tqdm
from paramiko import SSHClient

from experiments import ROOT_DIR


def main(model='diehl_and_cook_2015', data='mnist', cluster='swarm2', param_string=None, match=None):
    """
    Downloads parameters for a particular network from the CICS swarm2 cluster.
    """
    f = os.path.join(ROOT_DIR, 'credentials.yml')

    try:
        creds = yaml.load(open(f, 'r'))
    except FileNotFoundError:
        print('Create "credentials.yml" in top-level folder with username, password attributes.')
        sys.exit()

    username = creds['username']
    password = creds['password']

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(f'{cluster}.cs.umass.edu', username=username, password=password)

    sftp = ssh.open_sftp()
    sftp.chdir(f'/mnt/nfs/work1/rkozma/{username}/experiments/params/{data}/{model}/')
    
    localpath = os.path.join(ROOT_DIR, 'params', data, model)
    if not os.path.isdir(localpath):
        os.makedirs(localpath, exist_ok=True)

    if param_string is None:
        for f in tqdm(sftp.listdir()):
            if match is not None:
                if f.startswith(match):
                    sftp.get(f, os.path.join(localpath, f))
            else:
                sftp.get(f, os.path.join(localpath, f))
    else:
        sftp.get(param_string + '.pt', os.path.join(localpath, param_string + '.pt'))
        sftp.get('auxiliary_' + param_string + '.pt', os.path.join(localpath, 'auxiliary_' + param_string + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--cluster', type=str, default='swarm2')
    parser.add_argument('--param_string', type=str, default=None)
    parser.add_argument('--match', type=str, default=None)
    args = parser.parse_args()

    model = args.model
    data = args.data
    cluster = args.cluster
    param_string = args.param_string
    match = args.match

    main(model, data, cluster, param_string, match)