# siamese_nn.py
# Matches A.12 from the appendix; defaults to 128-d KGE + 768-d text, float64, BCE+sigmoid.
import torch
import torch.nn as nn
import torch.nn.functional as F

# If you change Node2Vec dim or text model size, update these defaults or pass sizes to the classes.
DEFAULT_KGE_DIM = 128
DEFAULT_TXT_DIM = 768

class Net(nn.Module):
    """Single-branch MLP on concatenated [KGE ⊕ TEXT] (not used by our trainer by default)."""
    def __init__(self, in_dim=DEFAULT_KGE_DIM + DEFAULT_TXT_DIM, dtype=torch.float64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 400, dtype=dtype)
        self.fc2 = nn.Linear(400, 150, dtype=dtype)
        self.fc3 = nn.Linear(150, 64,  dtype=dtype)
        self.fc4 = nn.Linear(64,  1,   dtype=dtype)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class UpgradedNet(nn.Module):
    """Single-branch MLP with BatchNorm + Dropout (also not used by default)."""
    def __init__(self, in_dim=DEFAULT_KGE_DIM + DEFAULT_TXT_DIM, dtype=torch.float64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 400, dtype=dtype); self.bn1 = nn.BatchNorm1d(400, dtype=dtype)
        self.fc2 = nn.Linear(400, 150, dtype=dtype);    self.bn2 = nn.BatchNorm1d(150, dtype=dtype)
        self.fc3 = nn.Linear(150, 64,  dtype=dtype);    self.bn3 = nn.BatchNorm1d(64,  dtype=dtype)
        self.fc4 = nn.Linear(64,  1,   dtype=dtype)
        self.dropout_less = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout_less(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x

class SiameseNetwork(nn.Module):
    """
    Two-branch (Siamese-ish) network like A.12:
      - branch_short: processes KGE (DEFAULT_KGE_DIM)
      - branch_text:  processes TEXT (DEFAULT_TXT_DIM)
      - concat -> MLP -> sigmoid
    Our train/infer scripts split inputs into (128, 768) chunks and call this.
    """
    def __init__(self, kge_dim=DEFAULT_KGE_DIM, txt_dim=DEFAULT_TXT_DIM, dtype=torch.float64):
        super().__init__()
        # KGE branch
        self.branch_kge = nn.Sequential(
            nn.Linear(kge_dim, 300, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm1d(300, dtype=dtype),
            nn.Linear(300, 250, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm1d(250, dtype=dtype),
            nn.Linear(250, 200, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm1d(200, dtype=dtype),
        )
        # TEXT branch
        self.branch_txt = nn.Sequential(
            nn.Linear(txt_dim, 300, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm1d(300, dtype=dtype),
            nn.Linear(300, 250, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm1d(250, dtype=dtype),
            nn.Linear(250, 200, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm1d(200, dtype=dtype),
        )
        # Head (post-concat)
        self.fc1 = nn.Linear(200 + 200, 128, dtype=dtype)
        self.fc2 = nn.Linear(128, 64, dtype=dtype)
        self.fc3 = nn.Linear(64, 1, dtype=dtype)
        self.sigmoid = nn.Sigmoid()
        self.dropout_less = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)

    def _head(self, x):
        x = self.dropout_less(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def forward(self, input_kge, input_txt):
        """
        input_kge: [B, kge_dim]
        input_txt: [B, txt_dim]
        returns:   [B, 1] probabilities
        """
        out_kge = self.branch_kge(input_kge)
        out_txt = self.branch_txt(input_txt)
        x = torch.cat((out_kge, out_txt), dim=1)
        return self._head(x)
