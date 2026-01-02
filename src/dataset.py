import os
import numpy as np
import torch
from torch.utils.data import Dataset


class XDViolenceDataset(Dataset):
    def __init__(self, root_dir, split="train", max_timesteps=100):
        self.root_dir = root_dir
        self.split = split
        self.max_timesteps = max_timesteps

        self.audio_dir = os.path.join(root_dir, "vggish-features", split)
        self.rgb_dir = os.path.join(root_dir, "i3d-rgb-features", split)
        self.flow_dir = os.path.join(root_dir, "i3d-flow-features", split)

        self.video_ids = [
            f.replace("__vggish.npy", "")
            for f in os.listdir(self.audio_dir)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.video_ids)

    def _load_feature(self, path):
        feat = np.load(path)
        T, D = feat.shape

        if T >= self.max_timesteps:
            feat = feat[:self.max_timesteps]
        else:
            pad = np.zeros((self.max_timesteps - T, D))
            feat = np.concatenate([feat, pad], axis=0)

        return torch.tensor(feat, dtype=torch.float32)

    def _parse_label(self, video_id):
        # video_id içinde "...label_XXX" var
        label_part = video_id.split("label_")[-1]

        if label_part.startswith("A"):
            return torch.tensor(0.0)
        else:
            return torch.tensor(1.0)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        audio_path = os.path.join(self.audio_dir, f"{video_id}__vggish.npy")
        rgb_path = os.path.join(self.rgb_dir, f"{video_id}__rgb.npy")
        flow_path = os.path.join(self.flow_dir, f"{video_id}__flow.npy")

        audio_feat = self._load_feature(audio_path)
        rgb_feat = self._load_feature(rgb_path)
        flow_feat = self._load_feature(flow_path)

        label = self._parse_label(video_id)

        return {
            "audio": audio_feat,
            "rgb": rgb_feat,
            "flow": flow_feat,
            "label": label
        }
