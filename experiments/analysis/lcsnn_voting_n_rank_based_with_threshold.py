import os
import torch

from tqdm import tqdm

location = 'cpu'

n = 3

def get_votes(spikes, scores):
    votes = torch.zeros(10)
    for j in range(9):
        patch = spikes[j * 100:(j + 1) * 100]
        sorted_spikes, sorted_indices = torch.sort(patch, descending=True)
        for index, s in enumerate(sorted_spikes[0:n]):
            if s != 0:
                selected_neuron_scores = scores[:, j * 100 + (patch == s).nonzero()]
                m_spikes, m_sort_ind = torch.sort(selected_neuron_scores.view(-1), descending=True)
                for mindex, m in enumerate(m_spikes):
                    if m > 4:
                        for i in (selected_neuron_scores == m).nonzero():
                            votes[i[0]] += (n - index)
    return votes

def main():
    # path = os.path.join('/media/bigdrive2/Devdhar/train')
    path = os.path.join('/home/hananel/git repo/Data/train')



    neuron_scores_per_class = torch.zeros([10, 900])
    for i in tqdm(range(227, 240)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        summed_spikes = torch.sum(spikes, dim=1)
        for k, example in enumerate(summed_spikes):
            for j in range(9):
                patch = example[j*100:(j+1)*100]
                max_spikes = torch.max(patch)
                if max_spikes != 0:
                    neuron_scores_per_class[int(labels[k]), j*100 + (patch == max_spikes).nonzero()] += 1

    all_labels = torch.LongTensor()
    all_predictions = torch.LongTensor()
    for i in tqdm(range(227, 240)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        summed_spikes = torch.sum(spikes, dim=1)
        predictions = torch.zeros(summed_spikes.shape[0])
        for k, example in enumerate(summed_spikes):
            votes = get_votes(example, neuron_scores_per_class)
            predictions[k] = torch.argmax(votes)
        all_labels = torch.cat([all_labels, labels.long()])
        all_predictions = torch.cat([all_predictions, predictions.long()])

    accuracy = (all_labels == all_predictions).float().mean() * 100
    print(f'Training accuracy: {accuracy:.2f}')


    # path = os.path.join('/media/bigdrive2/Devdhar/test')
    path = os.path.join('/home/hananel/git repo/Data/test')


    all_labels = torch.LongTensor()
    all_predictions = torch.LongTensor()
    for i in tqdm(range(1, 40)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        summed_spikes = torch.sum(spikes, dim=1)
        predictions = torch.zeros(summed_spikes.shape[0])
        for k, example in enumerate(summed_spikes):
            votes = get_votes(example, neuron_scores_per_class)
            predictions[k] = torch.argmax(votes)
        all_labels = torch.cat([all_labels, labels.long()])
        all_predictions = torch.cat([all_predictions, predictions.long()])

    accuracy = (all_labels == all_predictions).float().mean() * 100
    print(f'Test accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    main()
