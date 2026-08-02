from pathlib import Path
import random

from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from presence_dataset import PresenceDataset
from presence_model import PresenceNet


# =========================
# 訓練設定
# =========================

SEED = 42

DATA_ROOT = Path("data/presence_samples")

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"

OUTPUT_PATH = Path("data/meta/best_presence_avgmax_balanced_hardP.pt")

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PRESENCE_THRESHOLD = 0.6


# =========================
# 固定隨機種子
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 計算二分類統計
# =========================

def update_confusion_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    counts: dict,
) -> None:
    """
    logits: (B,)
    labels: (B,), 0=absent, 1=present
    """

    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    targets = labels >= 0.5

    counts["tp"] += int(
        (predictions & targets).sum().item()
    )

    counts["tn"] += int(
        ((~predictions) & (~targets)).sum().item()
    )

    counts["fp"] += int(
        (predictions & (~targets)).sum().item()
    )

    counts["fn"] += int(
        ((~predictions) & targets).sum().item()
    )


def calculate_metrics(counts: dict) -> dict:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # absent 被誤判成 present
    false_positive_rate = fp / max(fp + tn, 1)

    # present 被誤判成 absent
    false_negative_rate = fn / max(fn + tp, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# =========================
# 訓練一個 epoch
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()

    total_loss = 0.0
    total_samples = 0

    counts = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }

    for batch in loader:
        # batch["frames"]: (B, 3, 8, 192, 192)
        # batch["label"]:  (B,)
        frames = batch["frames"].to(
            device=device,
            dtype=torch.float32,
        )

        labels = batch["label"].to(
            device=device,
            dtype=torch.float32,
        )

        # 清除上一個 batch 累積的梯度
        optimizer.zero_grad()

        # forward
        logits = model(frames)

        # 計算 loss
        loss = criterion(logits, labels)

        # backward
        loss.backward()

        # 更新模型參數
        optimizer.step()

        batch_size = frames.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        update_confusion_counts(
            logits=logits.detach(),
            labels=labels,
            threshold=PRESENCE_THRESHOLD,
            counts=counts,
        )

    average_loss = total_loss / max(total_samples, 1)
    metrics = calculate_metrics(counts)

    return {
        "loss": average_loss,
        **metrics,
    }


# =========================
# 驗證或測試
# =========================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    counts = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }

    for batch in loader:
        frames = batch["frames"].to(
            device=device,
            dtype=torch.float32,
        )

        labels = batch["label"].to(
            device=device,
            dtype=torch.float32,
        )

        logits = model(frames)
        loss = criterion(logits, labels)

        batch_size = frames.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        update_confusion_counts(
            logits=logits,
            labels=labels,
            threshold=PRESENCE_THRESHOLD,
            counts=counts,
        )

    average_loss = total_loss / max(total_samples, 1)
    metrics = calculate_metrics(counts)

    return {
        "loss": average_loss,
        **metrics,
    }


# =========================
# 顯示統計
# =========================

def print_epoch_result(
    epoch: int,
    train_result: dict,
    val_result: dict,
) -> None:
    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        "Train | "
        f"loss={train_result['loss']:.4f} "
        f"acc={train_result['accuracy']:.3f} "
        f"precision={train_result['precision']:.3f} "
        f"recall={train_result['recall']:.3f} "
        f"fpr={train_result['fpr']:.3f} "
        f"fnr={train_result['fnr']:.3f}"
    )

    print(
        "Val   | "
        f"loss={val_result['loss']:.4f} "
        f"acc={val_result['accuracy']:.3f} "
        f"precision={val_result['precision']:.3f} "
        f"recall={val_result['recall']:.3f} "
        f"fpr={val_result['fpr']:.3f} "
        f"fnr={val_result['fnr']:.3f}"
    )

    print(
        "Val confusion | "
        f"TP={val_result['tp']} "
        f"TN={val_result['tn']} "
        f"FP={val_result['fp']} "
        f"FN={val_result['fn']}"
    )

def build_balanced_sampler(
    dataset: PresenceDataset,
) -> WeightedRandomSampler:
    groups = [
        (exposure, label)
        for exposure, label in zip(
            dataset.exposures,
            dataset.labels,
        )
    ]

    group_counts = Counter(groups)

    print("\nTrain group counts:")

    for group, count in sorted(group_counts.items()):
        exposure, label = group
        class_name = "present" if label == 1 else "absent"

        print(
            f"  {exposure}/{class_name}: {count}"
        )

    sample_weights = [
        1.0 / group_counts[group]
        for group in groups
    ]

    return WeightedRandomSampler(
        weights=torch.tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(dataset),
        replacement=True,
    )

# =========================
# 主程式
# =========================

def main() -> None:
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device: {device}")

    # 直接讀取已經切分好的資料夾
    train_dataset = PresenceDataset(TRAIN_DIR)
    val_dataset = PresenceDataset(VAL_DIR)
    test_dataset = PresenceDataset(TEST_DIR)

    print(f"train samples: {len(train_dataset)}")
    print(f"validation samples: {len(val_dataset)}")
    print(f"test samples: {len(test_dataset)}")

    train_sampler = build_balanced_sampler(
        train_dataset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = PresenceNet().to(device)

    # 二分類：
    # label=0 absent
    # label=1 present
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_epoch = -1

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, EPOCHS + 1):
        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_result = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print_epoch_result(
            epoch=epoch,
            train_result=train_result,
            val_result=val_result,
        )

        # validation loss 最低時存模型
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_result["loss"],
                    "val_metrics": val_result,
                    "threshold": PRESENCE_THRESHOLD,
                },
                OUTPUT_PATH,
            )

            print(
                f"[checkpoint] saved best model "
                f"at epoch={epoch}, "
                f"val_loss={best_val_loss:.4f}"
            )

    print(
        f"\nTraining finished. "
        f"Best epoch={best_epoch}, "
        f"best val loss={best_val_loss:.4f}"
    )

    # 重新載入 validation 表現最佳的模型
    checkpoint = torch.load(
        OUTPUT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # test 只在所有訓練和選模完成後跑一次
    test_result = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n===== Final Test Result =====")

    print(
        f"loss={test_result['loss']:.4f} "
        f"acc={test_result['accuracy']:.3f} "
        f"precision={test_result['precision']:.3f} "
        f"recall={test_result['recall']:.3f} "
        f"fpr={test_result['fpr']:.3f} "
        f"fnr={test_result['fnr']:.3f}"
    )

    print(
        "Test confusion | "
        f"TP={test_result['tp']} "
        f"TN={test_result['tn']} "
        f"FP={test_result['fp']} "
        f"FN={test_result['fn']}"
    )


if __name__ == "__main__":
    main()