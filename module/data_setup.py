import regex
import torch
import os
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset

class Vocabulary:
    def __init__(self, vocab_file, seq_length):
        self.vocab_file = vocab_file
        self.seq_length = seq_length
        self.characters = []
        self.string_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2, " ": 3}
        self.index_to_string = {0: "<pad>", 1: "<start>", 2: "<end>", 3: " "}

    def build_vocab(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                _, label = line.rstrip().split('--------')
                tokens = regex.findall(r'\X', label)
                for token in tokens:
                    if token != " ":
                        self.characters.append(token)
            f.close()
        for index, token in enumerate(set(self.characters)):
            self.string_to_index[token] = index + 4
            self.index_to_string[index + 4] = token

    def vectorize_text(self, sentence, add_special_token=True):
        tokens = regex.findall(r'\X', sentence)
        vectors = []
        if add_special_token:
            vectors.append(self.string_to_index["<start>"])
        for token in tokens:
            vectors.append(self.string_to_index[token])
        if add_special_token:
            vectors.append(self.string_to_index["<end>"])
        n = self.seq_length - len(vectors)
        if n > 0:
            vectors.extend([self.string_to_index["<pad>"] for _ in range(n)])
        return vectors

    def convert_text(self, vectors):
        texts = [self.index_to_string[vector] for vector in vectors]
        return texts

class CustomDataset(Dataset):
    def __init__(self, folder_name, file_path, vocab_file, seq_length, image_size):
        self.folder_name = folder_name
        self.file_path = file_path
        self.vocab_file = vocab_file
        self.seq_length = seq_length
        self.image_size = image_size
        # Get input and output
        self.input = []
        self.output = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                image_name, label = line.rstrip().split('--------')
                self.input.append(image_name)
                self.output.append(label)
            f.close()
        # Build vocabulary.txt
        self.vocab = Vocabulary(self.vocab_file, self.seq_length)
        self.vocab.build_vocab()
        # Transform
        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float)
        ])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        # Get image
        path = self.input[index]
        image_path = os.path.join(self.folder_name, path)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        # Get label
        label = self.output[index]
        vector_label = self.vocab.vectorize_text(label)
        return image_tensor, torch.Tensor(vector_label).int()