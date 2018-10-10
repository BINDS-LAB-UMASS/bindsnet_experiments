import os

from experiments import ROOT_DIR


def test_scripts():
    # language=rst
    """
    Systematically tests all scripts in the ``scripts/`` directory.
    """
    for d in os.listdir(os.path.join(ROOT_DIR, 'experiments')):
        for s in os.listdir(os.path.join(ROOT_DIR, 'experiments', d)):
            if not s == '__init__.py' and s.endswith('.py'):
                pass