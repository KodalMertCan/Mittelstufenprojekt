import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class MultiViewPigDataset(Dataset):
    def __init__(self, group_indices, dataframe, transform, max_views=4,
                 img_size=224, is_train=True):
        self.group_ids = list(group_indices.keys())
        self.group_indices = group_indices
        self.df = dataframe
        self.transform = transform
        self.max_views = max_views
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        gid = self.group_ids[idx]
        indices = self.group_indices[gid]
        rows = self.df.iloc[indices]
        label = int(rows.iloc[0]['class_id'])

        crops = []
        for _, row in rows.iterrows():
            try:
                img = Image.open(row['img_path']).convert('RGB')
                crop = img.crop((
                    int(row['xmin']), int(row['ymin']),
                    int(row['xmax']), int(row['ymax'])
                ))
                crops.append(crop)
            except Exception:
                continue

        if len(crops) == 0:
            crops = [Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))]

        if len(crops) > self.max_views:
            if self.is_train:
                crops = random.sample(crops, self.max_views)
            else:
                crops = crops[:self.max_views]

        n_real = len(crops)

        while len(crops) < self.max_views:
            crops.append(random.choice(crops[:n_real]))

        views = torch.stack([self.transform(c) for c in crops])
        mask = torch.zeros(self.max_views)
        mask[:n_real] = 1.0

        return views, mask, torch.tensor(label, dtype=torch.long)
