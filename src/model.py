import torch
from torch import nn
import torch.nn.functional as F

class TemporalMeanPooling(nn.Module):
    def forward(self, x):
        return x.mean(dim=1) # (B, T, D) -> (B, D)
    
class AudioAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = TemporalMeanPooling()

        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

        def forward(self, x):
            x = self.pool(x)
            x = self.mlp(x)
            return x

class VisualAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = TemporalMeanPooling()

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )

    def forward(self, x): # (B, T, 1024) -> (B, 256)
        x = self.pool(x)
        x = self.mlp(x)
        return x
    
class MotionAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = TemporalMeanPooling()

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )

    def forward(self, x): # (B, T, 1024) -> (B, 256)
        x = self.pool(x)
        x = self.mlp(x)
        return x
    
class DecisionAgent(nn.Module):
    def __init__(self);
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

    def forward(self, f_audio, f_rgb, f_flow):
        fusion = torch.cat([f_audio, f_rgb, f_flow], dim=1)
        out = self.classifier(fusion)
        return out

class MultiAgentViolanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.AudioAgent = AudioAgent()
        self.VisualAgent = VisualAgent()
        self.MotionAgent = MotionAgent()
        self.DecisionAgent = DecisionAgent()

    def forward(self, audio, rgb, flow):

        f_audio = self.AudioAgent(audio)
        f_rgb = self.VisualAgent(rgb)
        f_flow = self.MotionAgent(flow)

        logits = self.DecisionAgent(f_audio, f_rgb, f_flow)
        return logits.squeeze(1)
    