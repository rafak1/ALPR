import torch.nn as nn
from . import utils

class CRNN(nn.Module):
    def __init__(self, nh):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((4, 1), (4, 1))
        )

        self.rnn = nn.Sequential(
            nn.LSTM(128, nh, bidirectional=False),
            #GetLSTMOutput(),
            #nn.LSTM(nh * 2, nh, bidirectional=False)
        )

        nclass = len(utils.alphabet) + 1
        self.embedding = nn.Linear(nh, nclass)

    def forward(self, x):
        #print(f"before {x.size()}")
        x = self.cnn(x)
        #print(f"after {x.size()}")
        x = x.squeeze(2).permute(2, 0, 1)  # [W, B, C]
        #print(f"after squeeze {x.size()} {x.dim()}")
        x, _ = self.rnn(x)
        x = self.embedding(x)
        #print(f"after embedding {x.size()} {x.dim()}")
        return x  # [W, B, nclass]
    

class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out