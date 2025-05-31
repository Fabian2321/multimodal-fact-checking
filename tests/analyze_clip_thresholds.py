import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def analyze_thresholds(csv_path, true_label_col='true_label', score_col='scores'):
    """
    Analyzes different thresholds for CLIP scores to find an optimal one.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    if score_col not in df.columns or true_label_col not in df.columns:
        print(f"Error: Required columns ('{score_col}', '{true_label_col}') not in CSV.")
        return

    print("--- Score Statistics by True Label ---")
    if df[true_label_col].nunique() < 2:
        print("Warning: Not enough classes in true_label_col to perform detailed analysis.")
        print(f"Score statistics for all data (true_label: {df[true_label_col].unique()}):")
        print(df[score_col].describe())
    else:
        print("Scores for True Label == 0 (Real):")
        print(df[df[true_label_col] == 0][score_col].describe())
        print("\nScores for True Label == 1 (Fake):")
        print(df[df[true_label_col] == 1][score_col].describe())

    print("\n--- Threshold Analysis ---")
    
    # Define the range of thresholds to test
    # Based on observed scores (approx. 8 to 41), let's test 20 to 35
    thresholds = [round(t * 0.5, 1) for t in range(int(20.0*2), int(35.0*2) + 1)] # 20.0 to 35.0 in 0.5 steps

    best_f1_macro = 0
    best_threshold_f1_macro = None
    best_f1_fake = 0
    best_threshold_f1_fake = None

    print(f"{'Threshold':<10} | {'Acc.':<6} | {'P(R)':<6} | {'R(R)':<6} | {'F1(R)':<6} | {'P(F)':<6} | {'R(F)':<6} | {'F1(F)':<6} | {'F1Macro':<8}")
    print("-" * 80)

    y_true = df[true_label_col]

    if len(y_true) == 0:
        print("No data to analyze.")
        return

    for threshold in thresholds:
        # Apply inverted logic: score >= threshold means predicted label 1 (Fake)
        y_pred = df[score_col].apply(lambda x: 1 if x >= threshold else 0)

        accuracy = accuracy_score(y_true, y_pred)
        
        # average=None returns per-class scores. We assume class 0 is 'Real', class 1 is 'Fake'.
        # If only one class is present in predictions, metrics might be tricky.
        # We specify labels=[0, 1] to ensure metrics are always reported for both, even if one is not predicted.
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # Metrics for Real (class 0) and Fake (class 1)
        p_real, r_real, f1_real = precision[0], recall[0], f1[0]
        p_fake, r_fake, f1_fake = precision[1], recall[1], f1[1]
        
        print(f"{threshold:<10.1f} | {accuracy:<6.3f} | {p_real:<6.3f} | {r_real:<6.3f} | {f1_real:<6.3f} | {p_fake:<6.3f} | {r_fake:<6.3f} | {f1_fake:<6.3f} | {macro_f1:<8.3f}")

        if macro_f1 > best_f1_macro:
            best_f1_macro = macro_f1
            best_threshold_f1_macro = threshold
        
        if f1_fake > best_f1_fake:
            best_f1_fake = f1_fake
            best_threshold_f1_fake = threshold

    print("\n--- Optimal Thresholds ---")
    print(f"Best Threshold for Macro F1-Score: {best_threshold_f1_macro} (F1-Macro: {best_f1_macro:.4f})")
    print(f"Best Threshold for F1-Score (Fake Class): {best_threshold_f1_fake} (F1-Fake: {best_f1_fake:.4f})")

if __name__ == '__main__':
    # Path to your CLIP results CSV
    csv_file_path = 'results/clip/clip_test_detailed_report/all_model_outputs.csv' # Adjusted path relative to project root
    analyze_thresholds(csv_file_path) 