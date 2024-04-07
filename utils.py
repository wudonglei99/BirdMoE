from torch.utils.data import dataset
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from typing import Tuple
import torch
from torch import nn, Tensor
from data import MyDataset

def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def prepare_data(dataset_name=None, input_size=1, bptt=1):
    all_datasets = ['wikitext2', 'wikitext103']
    assert dataset_name.lower() in all_datasets, "dataset must in {}".format(all_datasets)
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    
    if dataset_name.lower() == 'wikitext2':
        from torchtext.datasets import WikiText2
        train_iter = WikiText2(root='/home/ywh/data', split='train')
        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        train_iter, val_iter, test_iter = WikiText2(root='/home/ywh/data')
    elif dataset_name.lower() == 'wikitext103':
        from torchtext.datasets import WikiText103
        train_iter = WikiText103(root='/home/ywh/data', split='train')
        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        train_iter, val_iter, test_iter = WikiText103(root='/home/ywh/data')
        
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)
    
    train_data = batchify(train_data, input_size)
    val_data = batchify(val_data, input_size)
    test_data = batchify(test_data, input_size)
    
    train_data_source = []
    train_data_target = []
    for i in range(0, train_data.size(0) - 1, bptt):
        data, targets = get_batch(train_data, i, bptt)
        train_data_source.append(torch.squeeze(data))
        train_data_target.append(torch.squeeze(targets))
        # print(data.shape, targets.shape)
    train_dataset = MyDataset(train_data_source, train_data_target)
    
    val_data_source = []
    val_data_target = []
    for i in range(0, val_data.size(0) - 1, bptt):
        data, targets = get_batch(val_data, i, bptt)
        val_data_source.append(torch.squeeze(data))
        val_data_target.append(torch.squeeze(targets))
    val_dataset = MyDataset(val_data_source, val_data_target)
    
    test_data_source = []
    test_data_target = []
    for i in range(0, test_data.size(0) - 1, bptt):
        data, targets = get_batch(test_data, i, bptt)
        test_data_source.append(torch.squeeze(data))
        test_data_target.append(torch.squeeze(targets))
    test_dataset = MyDataset(test_data_source, test_data_target)
    
    
    
    data_dict = {
        'train_dataset':   train_dataset,
        'val_dataset':     val_dataset,
        'test_dataset':    test_dataset,
        'vocab':        vocab,
    }
    
    return data_dict



def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target





def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
