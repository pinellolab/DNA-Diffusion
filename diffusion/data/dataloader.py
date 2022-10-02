from torch.utils.data import Dataset



class SequenceDataset(Dataset):
    "SequenceDataset that includes classes for conditonal generation"

    def __init__(self, seqs, classes=None, transform=None):
        self.seqs = seqs
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        image = self.seqs[index]

        x = self.transform(image)

        y = None
        if self.classes is not None:
            y = self.classes[index]
        
        return x, y
  