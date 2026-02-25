import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MyCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, use_attention=False):
        super().__init__()
        self.cnn = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        ).features
        # קפיאת 2 שכבות ראשונות
        for i, layer in enumerate(self.cnn):
            if i < 4:
                for p in layer.parameters():
                    p.requires_grad = False

        self.projection = nn.Linear(576, hidden_size)
        self.rnn = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.3
        )
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(
                hidden_size * 2, num_heads=8, batch_first=True
            )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)          # Gray → RGB
        feat = self.cnn(x)                  # [B,576,1,W']
        feat = feat.squeeze(2).permute(0, 2, 1)   # [B,W',576]
        feat = self.dropout(self.projection(feat)) # [B,W',256]
        out, _ = self.rnn(feat)             # [B,W',512]
        if self.use_attention:
            out, _ = self.attn(out, out, out)
        return self.fc(out)                 # [B,W',num_classes]
