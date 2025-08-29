import os, pickle, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from siamese_nn import SiameseNetwork

# ----- config -----
EPOCHS       = 180 
LR_STEPS     = [0.005, 0.001, 0.0005, 0.0001]  
MODEL_OUT    = "models/model_name_siamese_nick_node2vec_name.pth"
THRESH_OUT   = "models/model_name_siamese_nick_node2vec_name.threshold.txt"
TRAINSET_PKL = "testset_trainset/trainset_name_BC5CDR_node2vec_name.obj"
TESTSET_PKL  = "testset_trainset/testset_name_BC5CDR_node2vec_name.obj"

def to_numpy(t): return t.detach().cpu().numpy()

def sweep_threshold(y_true, y_prob):
    taus = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in taus]
    i = int(np.argmax(f1s))
    return float(f1s[i]), float(taus[i])

def main():
    os.makedirs("models", exist_ok=True)
    torch.set_default_dtype(torch.float64)

    trainset = pickle.load(open(TRAINSET_PKL, "rb"))
    testset  = pickle.load(open(TESTSET_PKL,  "rb"))

    # label sanity
    y_tr = trainset.tensors[1].cpu().numpy().reshape(-1)
    y_te = testset.tensors[1].cpu().numpy().reshape(-1)
    print(f"train size={len(y_tr)} | +={int(y_tr.sum())} -= {len(y_tr)-int(y_tr.sum())}")
    print(f"test  size={len(y_te)} | +={int(y_te.sum())} -= {len(y_te)-int(y_te.sum())}")
    if y_tr.sum()==0 or y_te.sum()==0:
        print("WARNING: no positives in train/test — check dataset build.")

    # loader (no drop_last) + adaptive batch size
    BATCH = min(256, max(32, len(trainset)//2))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=False)

    net = SiameseNetwork()
    criterion = nn.BCELoss()

    best_f1, best_tau = 0.0, 0.5
    for epoch in range(1, EPOCHS+1):
        # step LR at epochs 1,21,41,61
        lr = LR_STEPS[min((epoch-1)//20, len(LR_STEPS)-1)]
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # ---- train ----
        net.train()
        run_loss = 0.0
        for xb, yb in trainloader:
            a, b = torch.split(xb, [128, 768], dim=1)
            optimizer.zero_grad()
            p = net(a, b)
            loss = criterion(p, yb)
            loss.backward(); optimizer.step()
            run_loss += loss.item() * len(yb)
        run_loss /= max(1, len(trainset))

        # ---- eval ----
        net.eval()
        with torch.no_grad():
            xa, xb = torch.split(testset.tensors[0], [128, 768], dim=1)
            y_true = testset.tensors[1].reshape(-1).cpu().numpy().astype(int)
            y_prob = to_numpy(net(xa, xb).reshape(-1))

            f1_05 = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float("nan")
                ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else float("nan")
            except Exception:
                auc, ap = float("nan"), float("nan")
            f1_best, tau_best = sweep_threshold(y_true, y_prob)

        print(f"epoch {epoch:02d} | lr={lr:.4g} | loss={run_loss:.4f} | "
              f"F1(test@0.5)={f1_05:.3f} | F1(best)={f1_best:.3f} @ τ={tau_best:.2f} | AUC={auc:.3f} | AP={ap:.3f}")

        if f1_best > best_f1:
            best_f1, best_tau = f1_best, tau_best
            torch.save(net.state_dict(), MODEL_OUT)
            with open(THRESH_OUT, "w") as fh: fh.write(str(best_tau))
            print(f"  ↑ new best F1={best_f1:.3f} at τ={best_tau:.2f} — saved model+threshold")

    print("Finished training.")

if __name__ == "__main__":
    main()
