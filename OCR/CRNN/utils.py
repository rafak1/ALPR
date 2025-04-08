import torch

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0 = blank CTC
idx_to_char = {v: k for k, v in char_to_idx.items()}

def encode_label(label):
    return [char_to_idx[c] for c in label]

def decode_label(encoded_label):
    return "".join([idx_to_char[idx.item()] for idx in encoded_label if idx != 0])

def decode_output(pred):
    preds = pred.argmax(2)
    preds = preds.permute(1, 0)

    decoded_strings = []

    for pred in preds:
        string = ""
        prev_char = None
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev_char:
                string += idx_to_char[idx]
            prev_char = idx
        decoded_strings.append(string)

    return decoded_strings

def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    return torch.stack(imgs), list(labels), list(lengths)