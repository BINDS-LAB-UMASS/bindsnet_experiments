import torch
import matplotlib.pyplot as plt

from foolbox.models import Model
from bindsnet.encoding import poisson
from bindsnet.evaluation import ngram
from bindsnet.analysis.plotting import plot_input

axes = None
ims = None


class BindsNETModel(Model):
    # language=rst
    """
    Foolbox ``Model`` wrapper for BindsNET spiking neural networks.
    """

    def __init__(self, model, bounds=(0, 255), channel_axis=1, preprocessing=(0, 1), ngram_scores=None):
        self._model = model
        self._ngram_scores = ngram_scores

        super().__init__(
            bounds, channel_axis, preprocessing
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def bounds(self):
        return self._bounds

    def channel_axis(self):
        return self._channel_axis

    def predictions(self, image):
        global axes, ims

        self._model.run(
            inpts={'X': poisson(
                datum=torch.tensor(image), time=250, dt=1
            )}, time=250, dt=1
        )
        spike_record = self._model.monitors['Y_spikes'].get('s').t().unsqueeze(0)
        label = ngram(spike_record, self._ngram_scores, self.num_classes(), 2).numpy()[0]

        self._model.reset_()

        # axes, ims = plot_input(image.reshape(20, 20), image.reshape(20, 20), axes=axes, ims=ims)
        # plt.pause(0.05)

        print(label)

        return label

    def batch_predictions(self, images):
        spike_record = torch.zeros(len(images), 250, self._model.layers['Y'].n)
        for i, image in enumerate(images):
            self._model.run(
                inpts={'X': poisson(
                    datum=torch.tensor(image), time=250, dt=1
                )}, time=250, dt=1
            )
            self._model.reset_()

            spike_record[i] = self._model.monitors['Y_spikes'].get('s').t()

        labels = ngram(spike_record, self._ngram_scores, self.num_classes(), 2).numpy()
        return labels

    def num_classes(self):
        return 10
