# Simplified analysis script that calculates accuracy, FPR, and FNR

import pandas as pd
import argparse
import os
import glob
from collections import defaultdict # Import defaultdict


def analyze_file(file_path):
    """Analyze a single results file and return accuracy, FPR, FNR, and counts."""
    try:
        # Load the file
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)

        # --- Verify required columns ---
        required_cols = ['autograder_judgment', 'label']
        if not all(col in df.columns for col in required_cols):
            print(f"Error analyzing {filename}: Missing required columns ('autograder_judgment', 'label')")
            return None

        # Filter to valid judgments (PASS or FAIL) and valid labels
        valid_df = df[df['autograder_judgment'].isin(['PASS', 'FAIL']) & df['label'].isin(['PASS', 'FAIL'])].copy()
        total_valid = len(valid_df)

        if total_valid == 0:
            print(f"{filename}: No valid judgments and labels found")
            return None

        # --- Calculate TP, TN, FP, FN ---
        valid_df['is_tp'] = (valid_df['autograder_judgment'] == 'PASS') & (valid_df['label'] == 'PASS')
        valid_df['is_tn'] = (valid_df['autograder_judgment'] == 'FAIL') & (valid_df['label'] == 'FAIL')
        valid_df['is_fp'] = (valid_df['autograder_judgment'] == 'PASS') & (valid_df['label'] == 'FAIL')
        valid_df['is_fn'] = (valid_df['autograder_judgment'] == 'FAIL') & (valid_df['label'] == 'PASS')

        tp = valid_df['is_tp'].sum()
        tn = valid_df['is_tn'].sum()
        fp = valid_df['is_fp'].sum()
        fn = valid_df['is_fn'].sum()

        # --- Calculate Metrics ---
        accuracy = (tp + tn) / total_valid if total_valid > 0 else 0

        # False Positive Rate (FPR) = FP / (FP + TN) = FP / Actual Negatives
        actual_negatives = fp + tn
        fpr = fp / actual_negatives if actual_negatives > 0 else 0

        # False Negative Rate (FNR) = FN / (FN + TP) = FN / Actual Positives
        actual_positives = fn + tp
        fnr = fn / actual_positives if actual_positives > 0 else 0

        print(f"{filename}: FPR={fpr:.2%}, FNR={fnr:.2%} ({total_valid} valid examples)")
        # Optional: Print counts TP={tp}, TN={tn}, FP={fp}, FN={fn}

        # --- Calculate Per-Model Metrics (if column exists) ---
        per_model_counts = None
        if 'explainer_model' in valid_df.columns:
            per_model_counts = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0})
            grouped = valid_df.groupby('explainer_model')
            for model_name, group in grouped:
                model_tp = group['is_tp'].sum()
                model_tn = group['is_tn'].sum()
                model_fp = group['is_fp'].sum()
                model_fn = group['is_fn'].sum()
                model_total = len(group)

                per_model_counts[model_name] = {
                    'tp': model_tp,
                    'tn': model_tn,
                    'fp': model_fp,
                    'fn': model_fn,
                    'total': model_total
                }
                # Optional: Print per-model stats for this specific file
                # model_acc = (model_tp + model_tn) / model_total if model_total > 0 else 0
                # print(f"  - {model_name}: Acc={model_acc:.2%} ({model_total} examples)")


        # Return overall metrics and per-model counts
        result_data = {
            'filename': filename,
            'accuracy': accuracy,
            'fpr': fpr,
            'fnr': fnr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total_valid': total_valid
        }
        if per_model_counts:
            result_data['per_model_counts'] = dict(per_model_counts) # Convert back to dict for safety if needed later

        return result_data

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy, FPR, and FNR from evaluation results")
    parser.add_argument("--files", nargs='+', help="Specific CSV files to analyze (e.g., autograder_runs/run1.csv)")
    parser.add_argument("--runs-dir", default="autograder_runs",
                        help="Directory containing evaluation CSV files (default: autograder_runs)")
    parser.add_argument("--analyze-all", action="store_true",
                        help="Analyze all CSV files found in the specified --runs-dir")
    args = parser.parse_args()

    # Get files to analyze
    if args.files:
        file_paths = args.files
    elif args.analyze_all:
        runs_dir = args.runs_dir
        if not os.path.isdir(runs_dir):
            print(f"Error: Directory '{runs_dir}' not found.")
            return
        file_paths = glob.glob(os.path.join(runs_dir, '*.csv'))
    else:
        print("Please specify files to analyze with --files or use --analyze-all to process files in --runs-dir")
        return

    if not file_paths:
        print(f"No CSV files found to analyze in the specified location(s).")
        return

    # Analyze each file
    results_list = []
    # Use defaultdict to easily aggregate per-model counts across files
    overall_per_model_counts = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0})
    has_per_model_data = False # Flag to check if any file had the column

    for file_path in file_paths:
        if os.path.isfile(file_path):
            result_data = analyze_file(file_path)
            if result_data is not None:
                results_list.append(result_data)
                # Aggregate per-model counts if they exist for this file
                if 'per_model_counts' in result_data:
                    has_per_model_data = True
                    for model_name, counts in result_data['per_model_counts'].items():
                        overall_per_model_counts[model_name]['tp'] += counts['tp']
                        overall_per_model_counts[model_name]['tn'] += counts['tn']
                        overall_per_model_counts[model_name]['fp'] += counts['fp']
                        overall_per_model_counts[model_name]['fn'] += counts['fn']
                        overall_per_model_counts[model_name]['total'] += counts['total']
        else:
             print(f"Skipping non-file path: {file_path}")

    # Print overall summary if multiple files were successfully analyzed
    if len(results_list) > 0: # Changed from > 1 to show summary even for one file
        print("\n=== Overall Summary ===")

        # Calculate overall metrics by summing counts
        total_tp = sum(r['tp'] for r in results_list)
        total_tn = sum(r['tn'] for r in results_list)
        total_fp = sum(r['fp'] for r in results_list)
        total_fn = sum(r['fn'] for r in results_list)
        grand_total = total_tp + total_tn + total_fp + total_fn

        overall_accuracy = (total_tp + total_tn) / grand_total if grand_total > 0 else 0
        overall_actual_negatives = total_fp + total_tn
        overall_fpr = total_fp / overall_actual_negatives if overall_actual_negatives > 0 else 0
        overall_actual_positives = total_fn + total_tp
        overall_fnr = total_fn / overall_actual_positives if overall_actual_positives > 0 else 0

        print(f"Metrics across {len(results_list)} files ({grand_total} total examples):")
        print(f"  Overall Accuracy: {overall_accuracy:.2%}")
        print(f"  Overall FPR:      {overall_fpr:.2%}")
        print(f"  Overall FNR:      {overall_fnr:.2%}")

        # --- Print Overall Per-Model Summary ---
        if has_per_model_data:
            print("\n=== Overall Summary by Explainer Model ===")
            # Sort models alphabetically for consistent output
            sorted_models = sorted(overall_per_model_counts.keys())

            for model_name in sorted_models:
                counts = overall_per_model_counts[model_name]
                model_total_tp = counts['tp']
                model_total_tn = counts['tn']
                model_total_fp = counts['fp']
                model_total_fn = counts['fn']
                model_grand_total = counts['total'] # Use the summed total

                if model_grand_total > 0:
                    model_accuracy = (model_total_tp + model_total_tn) / model_grand_total
                    model_actual_negatives = model_total_fp + model_total_tn
                    model_fpr = model_total_fp / model_actual_negatives if model_actual_negatives > 0 else 0
                    model_actual_positives = model_total_fn + model_total_tp
                    model_fnr = model_total_fn / model_actual_positives if model_actual_positives > 0 else 0

                    print(f"\nModel: {model_name} ({model_grand_total} total examples)")
                    print(f"  Accuracy: {model_accuracy:.2%}")
                    print(f"  FPR:      {model_fpr:.2%}")
                    print(f"  FNR:      {model_fnr:.2%}")
                    # Optional: print counts
                    # print(f"  Counts: TP={model_total_tp}, TN={model_total_tn}, FP={model_total_fp}, FN={model_total_fn}")
                else:
                    print(f"\nModel: {model_name} (0 examples)")

        # Optional: Print sorted individual file results again
        # results_list.sort(key=lambda x: x['accuracy'], reverse=True)
        # print("\nIndividual File Results (sorted by accuracy):")
        # for r in results_list:
        #     print(f"  {r['filename']}: Acc={r['accuracy']:.2%}, FPR={r['fpr']:.2%}, FNR={r['fnr']:.2%}")

if __name__ == "__main__":
    main() 