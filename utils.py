from torch.utils.data import Dataset


def count_max_nodes(dataset: Dataset):
    c = 0
    for data in dataset:
        tmp = data.num_nodes
        c = c if c > tmp else tmp
    return c