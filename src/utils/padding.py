import torch

def pad_sequences(seqs, max_len):
    padded = []
    for s in seqs:
        if len(s) >= max_len:
            padded.append(s[:max_len])
        else:
            padded.append(s + [0] * (max_len - len(s)))  # 0 = <PAD>
    return torch.LongTensor(padded)
