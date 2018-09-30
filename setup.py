from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

version = '0.1'

setup(
    name='experiments',
    version=version,
    description='Experiments with BindsNET spiking neural network simulator',
    license='AGPL-3.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/djsaunde/bindsnet_experiments',
    author='Daniel Saunders',
    author_email='djsaunde@cs.umass.edu',
    packages=find_packages(),
    zip_safe=False,
    download_url='https://github.com/djsaunde/bindsnet_experiments/archive/%s.tar.gz' % version,
    install_requires=[
        'numpy>=1.14.2', 'torch>=0.4.1', 'matplotlib>=2.1.0', 'bindsnet>=0.1.7', 'PyYAML', 'tqdm', 'paramiko',
        'scikit-learn'
    ],
)
