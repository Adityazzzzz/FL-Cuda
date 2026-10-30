import argparse
import os
import csv
import torch
import numpy as np
import pandas as pd
from datasets import MIDataset, EA, ArrayToTensor
from model import EEGNet


def load_attack_results(model_dir, subjects):
    """
    Reads per-subject attack success rates from evaluation_results.csv if it exists.
    Returns a list aligned to `subjects`, or None if not available.
    """
    csv_path = os.path.join(model_dir, "evaluation_results.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0)
        if "Attack Success (%)" in df.index:
            row = df.loc["Attack Success (%)"]
            return [round(float(row[str(s)]), 2) if str(s) in row.index else None for s in subjects]
    except Exception:
        pass
    return None


def evaluate_models(model_dir, data_path="./data/DEAP", sub_ids="1,2,3,4,5,6,7,8,9",
                    global_epochs=None, local_epochs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = [int(i) for i in sub_ids.split(",")]

    accs = []
    evaluated_subs = []

    data_transform = [EA(), ArrayToTensor()]
    label_transform = [ArrayToTensor()]

    # Try to load cached attack results from a previous full training run
    cached_attack = load_attack_results(model_dir, subjects)

    print(f"\nEvaluating checkpoints in: {model_dir}")
    print(f"Device: {device}")
    if global_epochs is not None:
        print(f"Run config:  global_epochs={global_epochs}  local_epochs={local_epochs}")
    print("-" * 80)

    for idx, s in enumerate(subjects):
        model_file = os.path.join(model_dir, f"Model_ServerSub{s}.pth")
        if not os.path.exists(model_file):
            print(f"Warning: {model_file} not found, skipping subject {s}.")
            continue

        model = EEGNet(
            sample_rate=250, channels=32, F1=8, D=2, F2=16,
            time=8064, class_num=2, drop_out=0.5, bn_track=False
        ).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        model.eval()

        test_dataset = MIDataset(
            random_state=42, subject_id=[s], root=data_path,
            mode="all", data_transform=data_transform, label_transform=label_transform
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X).argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += len(y)

        acc = round(100.0 * correct / total, 2)
        accs.append(acc)
        evaluated_subs.append(s)

        att_str = ""
        if cached_attack and idx < len(cached_attack) and cached_attack[idx] is not None:
            att_str = f" | Attack Success = {cached_attack[idx]:.2f}%"
        print(f"  Subject {s}: Accuracy = {acc:.2f}% | Error Ratio = {100.0 - acc:.2f}%{att_str}")

    if not accs:
        print("No valid models found to evaluate.")
        return

    acc_arr = np.array(accs, dtype=float)
    mean_acc = round(float(np.mean(acc_arr)), 2)
    median_acc = round(float(np.median(acc_arr)), 2)

    error_ratios = np.round(100.0 - acc_arr, 2)
    mean_err = round(float(np.mean(error_ratios)), 2)
    median_err = round(float(np.median(error_ratios)), 2)

    columns = evaluated_subs + ["Mean", "Median"]
    acc_row    = list(acc_arr) + [mean_acc, median_acc]
    err_row    = list(error_ratios) + [mean_err, median_err]
    err_dec_row = [round(e / 100.0, 4) for e in err_row]

    rows  = [acc_row, err_row, err_dec_row]
    index = ["Test Acc (%)", "Error Ratio (%)", "Error Ratio (0-1)"]

    # Attach attack row if cached data aligns
    att_arr = None
    if cached_attack and len(cached_attack) == len(evaluated_subs):
        valid_att = [v for v in cached_attack if v is not None]
        if valid_att:
            att_arr = np.array(cached_attack, dtype=float)
            mean_att  = round(float(np.mean(att_arr)), 2)
            median_att = round(float(np.median(att_arr)), 2)
            att_row = list(att_arr) + [mean_att, median_att]
            rows.append(att_row)
            index.append("Attack Success (%)")

    df = pd.DataFrame(rows, columns=columns, index=index)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}" if abs(x) >= 1 else f"{x:.4f}")

    print("\n" + "=" * 80)
    print("                           EVALUATION SUMMARY")
    print("=" * 80)
    if global_epochs is not None:
        print(f"  global_epochs={global_epochs}  local_epochs={local_epochs}")
    print(f"  Subjects: {evaluated_subs}")
    print("-" * 80)
    print("\n[Per-Subject Matrix]")
    print(df.to_string())
    print("\n" + "-" * 80)
    print("Aggregate Metrics:")
    print(f"  Mean Accuracy:        {mean_acc:.2f}%")
    print(f"  Median Accuracy:      {median_acc:.2f}%")
    print(f"  Mean Error Ratio:     {mean_err:.2f}%  ({mean_err/100:.4f})")
    print(f"  Median Error Ratio:   {median_err:.2f}%  ({median_err/100:.4f})")
    if att_arr is not None:
        print(f"  Mean Attack Success:  {mean_att:.2f}%")
        print(f"  Median Attack Success:{median_att:.2f}%")
    print("=" * 80 + "\n")

    out_csv = os.path.join(model_dir, "evaluation_results.csv")
    df.to_csv(out_csv)
    print(f"Saved evaluation matrix to: {out_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Saved Model Checkpoints")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing Model_ServerSub*.pth")
    parser.add_argument("--data_path", type=str, default="./data/DEAP", help="Path to DEAP dataset")
    parser.add_argument("--sub_id", type=str, default="1,2,3,4,5,6,7,8,9", help="Comma-separated subject IDs")
    parser.add_argument("--global_epochs", type=int, default=None, help="(Optional) global epochs used in this run")
    parser.add_argument("--local_epochs", type=int, default=None, help="(Optional) local epochs used in this run")
    args = parser.parse_args()

    evaluate_models(args.model_dir, args.data_path, args.sub_id, args.global_epochs, args.local_epochs)
