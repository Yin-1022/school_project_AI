from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models import TeacherValueNet
from read_rollout_dataset import RolloutValueDataset

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "data/meta/best_value_net.pt"

def main():
    dataset = RolloutValueDataset("data/rollouts", gamma=0.99)
    total_size = len(dataset)

    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TeacherValueNet(in_ch=3, extra_dim=24).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    Path("data/meta").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0

        for frames, extra, target_return in train_loader:
            frames = frames.to(DEVICE)
            extra = extra.to(DEVICE)
            target_return = target_return.to(DEVICE)

            optimizer.zero_grad()
            pred_value = model(frames, extra)
            loss = criterion(pred_value, target_return)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * target_return.size(0)
            train_total += target_return.size(0)

        train_loss = train_loss_sum / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_total = 0

        with torch.no_grad():
            for frames, extra, target_return in val_loader:
                frames = frames.to(DEVICE)
                extra = extra.to(DEVICE)
                target_return = target_return.to(DEVICE)

                pred_value = model(frames, extra)
                loss = criterion(pred_value, target_return)

                val_loss_sum += loss.item() * target_return.size(0)
                val_total += target_return.size(0)

        val_loss = val_loss_sum / max(val_total, 1)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> best saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()