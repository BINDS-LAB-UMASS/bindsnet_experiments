from fastai import Learner
from fastai.metrics import error_rate
from fastai.vision.models import resnet34
from fastai.datasets import URLs, untar_data
from fastai.vision.data import ImageDataBunch, get_transforms

n_epochs = 1

path = untar_data(URLs.CIFAR, dest='../../data/cifar10')
data = ImageDataBunch.from_folder(
    path, train=f'train', valid=f'valid', ds_tfms=get_transforms(), size=224
)
learner = Learner(data, resnet34().cuda(), metrics=error_rate)
learner.fit_one_cycle(n_epochs)
