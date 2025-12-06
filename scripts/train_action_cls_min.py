# scripts/train_action_cls_min.py
import os, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from read_datasets import ClipDataset, CLASS_TO_ID
from models import Small3DNet
from pathlib import Path
from sklearn.metrics import classification_report


BASE_DIR = Path(__file__).resolve().parents[1]      # 專案根
CSV_PATH = (BASE_DIR / "data" / "meta" / "label.csv") if (BASE_DIR / "data" / "meta" / "label.csv").exists() else (BASE_DIR / "data" / "meta" / "label.csv")
ROOT = BASE_DIR / "data"  
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 8
NUM_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    seed_all(42)
    full = ClipDataset(str(CSV_PATH), root=str(ROOT), expected_frames=8, size=(192,192), augment=True)

    if len(full) == 0:
        print("[fatal] Dataset is empty after path resolution.")
        print("        1) 檢查 labels.csv 的 clip_path 是否以 'clips/...' 開頭")
        print("        2) ROOT 應設為 <專案根>/data，目前：", ROOT)
        print("        3) 實際檔案副檔名：若是 .avi，CSV 寫 .mp4 也行（讀取器會自動對應），但檔案必須存在")
        return


    # 簡單切 train/val：8:2
    indices = list(range(len(full)))
    labels_all = [y for _, y in full.items]
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels_all)

    from torch.utils.data import Subset
    train_set = Subset(full, train_idx)
    val_set   = Subset(full, val_idx)
    # val 不做增強
    val_set.dataset.augment = False

    # 類別權重（處理類別不平衡；若你分佈很均衡也可拿掉）
    counts = np.zeros(NUM_CLASSES, dtype=int)
    for _, y in train_set:
        counts[y.item()] += 1
    weights = counts.sum() / np.maximum(counts, 1)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = Small3DNet(in_ch=3, num_classes=NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    best_acc, best_path = 0.0, "data/meta/best_action_cls.pt"
    os.makedirs("data/meta", exist_ok=True)

    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {ep}/{EPOCHS}")
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward(); opt.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.3f}")

        # 驗證
        model.eval()
        v_total, v_correct = 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                pred = logits.argmax(1)
                v_total += y.size(0)
                v_correct += (pred == y).sum().item()
                all_pred.extend(pred.cpu().tolist())
                all_true.extend(y.cpu().tolist())
        acc = v_correct / max(v_total,1)
        print(f"[Val] acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print(f"  -> best saved to {best_path}")

    # 最後印一份分類報告（大致看哪類弱）
    print("\nValidation classification report:")
    target_names = list(CLASS_TO_ID.keys())
    if len(all_true) == 0:
        print("No validation samples were evaluated.")
    else:
        labels_present = sorted(set(all_true) | set(all_pred))
        names_present  = [target_names[i] for i in labels_present]
        print(classification_report(
            all_true, all_pred,
            labels=labels_present,
            target_names=names_present,
            zero_division=0
        ))
        missing = [target_names[i] for i in range(NUM_CLASSES) if i not in labels_present]
        if missing:
            print(f"[warn] absent in val set: {missing}")

if __name__ == "__main__":
    main()
