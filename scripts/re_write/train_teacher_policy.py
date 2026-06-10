from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from read_teacher_dataset import TeacherDataset
from models import TeacherPolicyNet, ExtraOnlyPolicyNet

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "data/meta/best_teacher_policy.pt"

def main():
    dataset = TeacherDataset("data/teacher_samples")
    total_size = len(dataset)

    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = TeacherPolicyNet(in_ch=3, extra_dim=24, num_actions=10).to(DEVICE)
    #model = ExtraOnlyPolicyNet(extra_dim=24, num_actions=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    Path("data/meta").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        # train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for frames, extra, action_id in train_loader:
            frames = frames.to(DEVICE)
            extra = extra.to(DEVICE)
            action_id = action_id.to(DEVICE)

            optimizer.zero_grad()
            logits = model(frames, extra)
            loss = criterion(logits, action_id)
            loss.backward()     #算梯度
            optimizer.step()    #更新模型

            train_loss += loss.item() * action_id.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == action_id).sum().item()
            train_total += action_id.size(0)
        
        train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, extra, action_id in val_loader:
                frames = frames.to(DEVICE)
                extra = extra.to(DEVICE)
                action_id = action_id.to(DEVICE)

                logits = model(frames, extra)
                loss = criterion(logits, action_id)

                val_loss += loss.item() * action_id.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == action_id).sum().item()
                val_total += action_id.size(0)
        
        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> best saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
