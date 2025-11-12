import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
import torch.optim as optim
from collections import Counter
import numpy as np


def read_sentences_from_file(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label_seq = []
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label_seq)
                    sentence = []
                    label_seq = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    char, tag = parts
                    sentence.append(char)
                    label_seq.append(tag)
        if sentence:
            sentences.append(sentence)
            labels.append(label_seq)
    return sentences, labels


class NERDataset(Dataset):
    def __init__(self, sentences, labels, char2idx, tag2idx):
        self.sentences = sentences
        self.labels = labels
        self.char2idx = char2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.labels[idx]

        char_ids = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in sentence]
        tag_ids = [self.tag2idx[t] for t in tags]
        return torch.tensor(char_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    chars, tags = zip(*batch)
    chars_pad = pad_sequence(chars, batch_first=True, padding_value=0)
    tags_pad = pad_sequence(tags, batch_first=True, padding_value=-1)  # -1 用于忽略
    mask = (chars_pad != 0)
    return chars_pad, tags_pad, mask


def load_pretrained_char_embeddings(char2idx, embedding_file, embed_dim=300):
    embeddings = np.random.normal(scale=0.1, size=(len(char2idx), embed_dim))
    embeddings[0] = 0
    found = 0
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embed_dim + 1:
                continue
            char = parts[0]
            if char in char2idx:
                vector = np.array(parts[1:], dtype='float32')
                embeddings[char2idx[char]] = vector
                found += 1
    print(f"Loaded {found} / {len(char2idx)} characters from pretrained embeddings.")
    return embeddings


class BiLSTM_CRF(nn.Module):
    def __init__(self, char2idx, embedding_file, tagset_size, embedding_dim, hidden_dim, pad_idx=0):
        super().__init__()
        pretrained_weight = load_pretrained_char_embeddings(char2idx, embedding_file,
                                                            embed_dim=embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_weight),
            freeze=False,
            padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, chars, tags=None, mask=None):
        embeds = self.embedding(chars)  # [B, L, E]
        lstm_out, _ = self.lstm(embeds)  # [B, L, H]
        emissions = self.hidden2tag(lstm_out)  # [B, L, T]

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            decoded = self.crf.decode(emissions, mask=mask)
            return decoded



EMBEDDING_FILE = r'/Volumes/My Passport/dataset/models/pretrained/sgns.wiki.char'


def train_model():
    train_file = 'data/train.txt'
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'bilstm_crf.pth')

    print("Loading training data...")
    train_sents, train_labels = read_sentences_from_file(train_file)
    char_counter = Counter(char for sent in train_sents for char in sent)
    tag_counter = Counter(tag for tags in train_labels for tag in tags)

    char2idx = {'<PAD>': 0, '<UNK>': 1}
    for char in char_counter:
        if char_counter[char] >= 1:
            char2idx[char] = len(char2idx)
    print("Sample chars in vocab:", list(char2idx.keys())[2:12])
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tag_counter))}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    print(f"Vocab size: {len(char2idx)}, Tag size: {len(tag2idx)}")

    train_dataset = NERDataset(train_sents, train_labels, char2idx, tag2idx)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"device is {device}")
    model = BiLSTM_CRF(
        char2idx=char2idx,
        embedding_file=EMBEDDING_FILE,
        tagset_size=len(tag2idx),
        embedding_dim=300,
        hidden_dim=64
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for chars, tags, mask in train_loader:
            chars, tags, mask = chars.to(device), tags.to(device), mask.to(device)
            loss = model(chars, tags, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': char2idx,
        'tag2idx': tag2idx,
        'idx2tag': idx2tag
    }, model_path)
    print(f"Model saved to {model_path}")


def test_model():
    model_path = 'model/bilstm_crf.pth'
    test_file = 'data/test.txt'

    if not os.path.exists(model_path):
        print("Model not found! Please run training first.")
        return

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    char2idx = checkpoint['char2idx']
    tag2idx = checkpoint['tag2idx']
    idx2tag = checkpoint['idx2tag']

    model = BiLSTM_CRF(
        char2idx=char2idx,
        embedding_file=EMBEDDING_FILE,
        tagset_size=len(tag2idx),
        embedding_dim=300,
        hidden_dim=128
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_sents, test_labels = read_sentences_from_file(test_file)
    test_dataset = NERDataset(test_sents, test_labels, char2idx, tag2idx)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for chars, tags, mask in test_loader:
            chars, mask = chars, mask
            preds = model(chars, mask=mask)
            for i in range(len(preds)):
                length = mask[i].sum().item()
                all_preds.extend(preds[i][:length])
                all_trues.extend(tags[i][:length].tolist())

    pred_tags = [idx2tag[idx] for idx in all_preds]
    true_tags = [idx2tag[idx] for idx in all_trues]

    correct = sum(p == t for p, t in zip(pred_tags, true_tags))
    accuracy = correct / len(pred_tags)
    def extract_entities(tags):
        entities = []
        i = 0
        while i < len(tags):
            if tags[i] == 'S':
                entities.append((i, i, 'ENT'))
                i += 1
            elif tags[i] == 'B':
                start = i
                i += 1
                while i < len(tags) and tags[i] == 'I':
                    i += 1
                end = i - 1
                entities.append((start, end, 'ENT'))
            else:
                i += 1
        return set(entities)

    true_entities = extract_entities(true_tags)
    pred_entities = extract_entities(pred_tags)

    tp = len(true_entities & pred_entities)
    fp = len(pred_entities - true_entities)
    fn = len(true_entities - pred_entities)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"Token-level Accuracy: {accuracy:.4f}")
    print(f"Entity-level Precision: {precision:.4f}")
    print(f"Entity-level Recall:    {recall:.4f}")
    print(f"Entity-level F1-score:  {f1:.4f}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python task2.py [train|test]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "train":
        train_model()
    elif mode == "test":
        test_model()
    else:
        print("Unknown mode. Use 'train' or 'test'.")


