from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models import TeacherActorCriticNet
from read_actor_critic_dataset import RolloutActorCriticDataset

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "data/meta/best_actor_critic_warmstart.pt"

POLICY_COEF = 1.0
VALUE_COEF = 0.5

def main():
    dataset = RolloutActorCriticDataset("data/rollouts", gamma=0.99)
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

    model = TeacherActorCriticNet(in_ch=3, extra_dim=24, num_actions=10).to(DEVICE)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    Path("data/meta").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_policy_loss_sum = 0.0
        train_value_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for frames, extra, action, target_return in train_loader:
            frames = frames.to(DEVICE)
            extra = extra.to(DEVICE)
            action = action.to(DEVICE)
            target_return = target_return.to(DEVICE)

            optimizer.zero_grad()

            logits, value = model(frames, extra)

            policy_loss = ce(logits, action)
            value_loss = mse(value, target_return)
            loss = POLICY_COEF * policy_loss + VALUE_COEF * value_loss

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * action.size(0)
            train_policy_loss_sum += policy_loss.item() * action.size(0)
            train_value_loss_sum += value_loss.item() * action.size(0)

            pred = logits.argmax(dim=1)
            train_correct += (pred == action).sum().item()
            train_total += action.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_policy_loss = train_policy_loss_sum / max(train_total, 1)
        train_value_loss = train_value_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_policy_loss_sum = 0.0
        val_value_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, extra, action, target_return in val_loader:
                frames = frames.to(DEVICE)
                extra = extra.to(DEVICE)
                action = action.to(DEVICE)
                target_return = target_return.to(DEVICE)

                logits, value = model(frames, extra)

                policy_loss = ce(logits, action)
                value_loss = mse(value, target_return)
                loss = POLICY_COEF * policy_loss + VALUE_COEF * value_loss

                val_loss_sum += loss.item() * action.size(0)
                val_policy_loss_sum += policy_loss.item() * action.size(0)
                val_value_loss_sum += value_loss.item() * action.size(0)

                pred = logits.argmax(dim=1)
                val_correct += (pred == action).sum().item()
                val_total += action.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_policy_loss = val_policy_loss_sum / max(val_total, 1)
        val_value_loss = val_value_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"(pi={train_policy_loss:.4f}, v={train_value_loss:.4f}) "
            f"train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} "
            f"(pi={val_policy_loss:.4f}, v={val_value_loss:.4f}) "
            f"val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> best saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()