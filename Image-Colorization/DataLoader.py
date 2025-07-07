from torch.utils.data import Dataset,DataLoader
from ColorizationDataset import ColorizationDataset

def make_dataloaders(batch_size=8, n_workers=4, pin_memory=True, **kwargs):
    
    
    dataset = ColorizationDataset(**kwargs)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    
    return dataloader 