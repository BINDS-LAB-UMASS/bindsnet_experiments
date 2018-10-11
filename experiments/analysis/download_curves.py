import os
import sys
import yaml
import argparse

from paramiko import SSHClient

from experiments import ROOT_DIR


def main(cluster='swarm2',
         model='diehl_and_cook_2015',
         data='mnist',
         param_string=None):
    """
    Downloads training curves for a particular network from a CICS cluster.
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
    sftp.chdir(f'/mnt/nfs/work1/rkozma/{username}/experiments/curves/{data}/{model}/')
    
    localpath = os.path.join(ROOT_DIR, 'curves', data, model)
    if not os.path.isdir(localpath):
        os.makedirs(localpath, exist_ok=True)

    sftp.get(param_string + '.pt', os.path.join(localpath, param_string + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str, default='swarm2')
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--param_string', type=str, required=True)
    args = parser.parse_args()

    cluster = args.cluster
    model = args.model
    data = args.data
    param_string = args.param_string

    main(cluster, model, data, param_string)