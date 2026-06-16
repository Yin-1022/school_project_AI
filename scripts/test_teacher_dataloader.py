from read_teacher_dataset import TeacherDataset
from torch.utils.data import DataLoader, random_split, Subset
import torch

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

    # train_end = int(0.75 * total_size)
    # val_start = int(0.80 * total_size)

    # train_indices = list(range(0, train_end))
    # val_indices = list(range(val_start, total_size))

    # train_set = Subset(dataset, train_indices)
    # val_set = Subset(dataset, val_indices)

    # DataLoader將繁雜的資料集（Dataset）轉換成易於模型訓練的 迭代器（Iterator）。
    # 負責自動資料分批（Batching）、隨機打散（Shuffling），支援多執行緒（Multi-process）平行載入
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    print(f"total: {total_size}")
    print(f"train: {len(train_set)}")
    print(f"val: {len(val_set)}")

    # iter() 主要用於將可迭代物件（如串列、元組、字典、字串）轉換為迭代器（Iterator）
    # 以便搭配 next() 函數逐一取得元素
    frames, extra, action_id = next(iter(train_loader))

    print("frames batch shape:", frames.shape)
    print("extra batch shape:", extra.shape)
    print("action_id batch shape:", action_id.shape)
    print("action_id dtype:", action_id.dtype)

if __name__ == "__main__":
    main()
