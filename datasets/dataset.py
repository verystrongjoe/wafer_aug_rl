from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, Xs, ys):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.Xs = Xs
        self.ys = ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return dict(x=self.Xs[idx], y=self.ys[idx], idx=idx)