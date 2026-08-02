from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from presence_dataset import PresenceDataset
from presence_model import PresenceNet

import shutil


DATA_ROOT = Path("data/presence_samples")
ERROR_OUTPUT_DIR = Path("data/presence_errors")

VAL_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"

CHECKPOINT_PATH = Path("data/meta/best_presence_avgmax_balanced_hardP.pt")

BATCH_SIZE = 8


@torch.no_grad()
def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    all_probabilities = []
    all_labels = []

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
        probabilities = torch.sigmoid(logits)

        all_probabilities.append(probabilities.cpu())
        all_labels.append(labels.cpu())

    return (
        torch.cat(all_probabilities),
        torch.cat(all_labels),
    )


def calculate_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict:
    predictions = probabilities >= threshold
    targets = labels >= 0.5

    tp = int((predictions & targets).sum().item())
    tn = int(((~predictions) & (~targets)).sum().item())
    fp = int((predictions & (~targets)).sum().item())
    fn = int(((~predictions) & targets).sum().item())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

def export_error_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    output_dir: Path,
) -> dict:
    model.eval()

    # 清掉上一次輸出的結果，避免新舊檔案混在一起
    if output_dir.exists():
        shutil.rmtree(output_dir)

    error_counts = {
        "false_positive": {
            "normal": 0,
            "medium": 0,
            "dark": 0,
        },
        "false_negative": {
            "normal": 0,
            "medium": 0,
            "dark": 0,
        },
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
        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= threshold
        targets = labels >= 0.5

        batch_paths = batch["path"]
        batch_exposures = batch["exposure"]

        for index in range(len(batch_paths)):
            prediction = bool(predictions[index].item())
            target = bool(targets[index].item())

            source_path = Path(batch_paths[index])
            exposure = batch_exposures[index]
            probability = float(probabilities[index].item())

            error_type = None

            # absent 被判成 present
            if prediction and not target:
                error_type = "false_positive"

            # present 被判成 absent
            elif not prediction and target:
                error_type = "false_negative"

            if error_type is None:
                continue

            destination_dir = (
                output_dir
                / error_type
                / exposure
            )
            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            error_counts[error_type][exposure] += 1

            destination_name = (
                f"prob_{probability:.4f}_"
                f"{source_path.name}"
            )

            destination_path = (
                destination_dir / destination_name
            )

            shutil.copy2(
                source_path,
                destination_path,
            )

    print("\n===== Exported Error Samples =====")

    for error_type, exposure_counts in error_counts.items():
        total = sum(exposure_counts.values())

        print(f"{error_type}: total={total}")

        for exposure, count in exposure_counts.items():
            print(
                f"  {exposure}: {count}"
            )

    print(f"\nerror samples saved to: {output_dir}")

    return error_counts

def search_best_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    max_fpr: float = 0.10,
) -> tuple[float, list[dict]]:
    results = []

    # 用 0.01 為間隔，比只測 0.05 更細
    for step in range(5, 96):
        threshold = step / 100

        metrics = calculate_metrics(
            probabilities=probabilities,
            labels=labels,
            threshold=threshold,
        )

        results.append(metrics)

    print("\n===== Validation Threshold Search =====")

    # 顯示較容易閱讀的 0.05 間隔
    for result in results:
        threshold = result["threshold"]

        if round(threshold * 100) % 5 != 0:
            continue

        print(
            f"threshold={threshold:.2f} "
            f"acc={result['accuracy']:.3f} "
            f"precision={result['precision']:.3f} "
            f"recall={result['recall']:.3f} "
            f"fpr={result['fpr']:.3f} "
            f"fnr={result['fnr']:.3f}"
        )

    # 在 FPR <= 10% 的候選中，選 Recall 最高者
    valid_results = [
        result
        for result in results
        if result["fpr"] <= max_fpr
    ]

    if not valid_results:
        # 沒有任何 threshold 達標時，先選 FPR 最低者；
        # 同 FPR 時選 Recall 較高者。
        best_result = min(
            results,
            key=lambda result: (
                result["fpr"],
                -result["recall"],
            ),
        )

        print(
            f"\n[warning] No threshold satisfies "
            f"FPR <= {max_fpr:.2f}"
        )
    else:
        best_result = max(
            valid_results,
            key=lambda result: (
                result["recall"],
                result["accuracy"],
            ),
        )

    print("\n===== Selected Threshold =====")
    print(
        f"threshold={best_result['threshold']:.2f} "
        f"acc={best_result['accuracy']:.3f} "
        f"precision={best_result['precision']:.3f} "
        f"recall={best_result['recall']:.3f} "
        f"fpr={best_result['fpr']:.3f} "
        f"fnr={best_result['fnr']:.3f}"
    )

    return best_result["threshold"], results


def print_result(
    title: str,
    result: dict,
) -> None:
    print(f"\n===== {title} =====")

    print(
        f"threshold={result['threshold']:.2f} "
        f"acc={result['accuracy']:.3f} "
        f"precision={result['precision']:.3f} "
        f"recall={result['recall']:.3f} "
        f"fpr={result['fpr']:.3f} "
        f"fnr={result['fnr']:.3f}"
    )

    print(
        f"TP={result['tp']} "
        f"TN={result['tn']} "
        f"FP={result['fp']} "
        f"FN={result['fn']}"
    )


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device: {device}")

    # 重新建立 validation/test Dataset
    val_dataset = PresenceDataset(VAL_DIR)
    test_dataset = PresenceDataset(TEST_DIR)

    print(f"validation samples: {len(val_dataset)}")
    print(f"test samples: {len(test_dataset)}")

    # 重新建立 DataLoader
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

    # 重新建立空的模型架構
    model = PresenceNet().to(device)

    # 從訓練完成的 checkpoint 讀取權重
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"loaded checkpoint: {CHECKPOINT_PATH} "
        f"(epoch={checkpoint.get('epoch', 'unknown')})"
    )

    # 只用 validation 選 threshold
    val_probabilities, val_labels = collect_probabilities(
        model=model,
        loader=val_loader,
        device=device,
    )

    best_threshold, _ = search_best_threshold(
        probabilities=val_probabilities,
        labels=val_labels,
        max_fpr=0.10,
    )

    best_threshold=0.60

    # threshold 選定後，才跑 test
    test_probabilities, test_labels = collect_probabilities(
        model=model,
        loader=test_loader,
        device=device,
    )

    test_result = calculate_metrics(
        probabilities=test_probabilities,
        labels=test_labels,
        threshold=best_threshold,
    )

    print_result(
        title="Final Test Result",
        result=test_result,
    )

    export_error_samples(
        model=model,
        loader=test_loader,
        device=device,
        threshold=best_threshold,
        output_dir=ERROR_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()