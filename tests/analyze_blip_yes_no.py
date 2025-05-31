import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def extract_blip_prediction(generated_text):
    """
    Extracts 'Yes.' or 'No.' from the beginning of the generated text
    and converts it to a binary label.
    'Yes.' (text matches image) -> maps to 0 (Real, consistent with Fakeddit 2_way_label)
    'No.'  (text mismatches image) -> maps to 1 (Fake, consistent with Fakeddit 2_way_label)
    Returns None if neither is found.
    """
    if not isinstance(generated_text, str):
        return None
    text_lower = generated_text.lower().strip()
    if text_lower.startswith("yes."):
        return 0 # 'Yes, text and image match' -> Real
    elif text_lower.startswith("no."):
        return 1 # 'No, text and image mismatch' -> Fake
    return None

def analyze_blip_outputs(csv_path, true_label_col='true_label', generated_text_col='generated_text', print_errors=True, num_errors_to_print=5):
    """
    Analyzes BLIP's Yes/No predictions from generated text.
    Optionally prints specific error cases where true_label is 1 (Fake) but predicted 0 (Real/Match).
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    if generated_text_col not in df.columns or true_label_col not in df.columns:
        print(f"Error: Required columns ('{generated_text_col}', '{true_label_col}') not in CSV.")
        return

    df['predicted_binary_label'] = df[generated_text_col].apply(extract_blip_prediction)

    eval_df = df.dropna(subset=['predicted_binary_label', true_label_col])
    eval_df['predicted_binary_label'] = eval_df['predicted_binary_label'].astype(int)
    eval_df[true_label_col] = eval_df[true_label_col].astype(int)

    y_true = eval_df[true_label_col]
    y_pred = eval_df['predicted_binary_label']

    if len(y_true) == 0:
        print("No valid predictions could be extracted for evaluation.")
        return

    print(f"--- BLIP Yes/No Prediction Analysis (Total samples evaluated: {len(y_true)}) ---")
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'], zero_division=0)

    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)
    
    yes_count = df[df['predicted_binary_label'] == 0].shape[0]
    no_count = df[df['predicted_binary_label'] == 1].shape[0]
    none_count = df['predicted_binary_label'].isnull().sum()
    print(f"\nPrediction Summary:")
    print(f"  - Predicted 'Yes.' (match, mapped to 0): {yes_count}")
    print(f"  - Predicted 'No.' (mismatch, mapped to 1): {no_count}")
    print(f"  - Could not extract Yes/No: {none_count}")

    if print_errors:
        # Find cases where true_label is 1 (Fake) but predicted_binary_label is 0 (Real/Match)
        error_cases = df[(df[true_label_col] == 1) & (df['predicted_binary_label'] == 0)]
        if not error_cases.empty:
            print(f"\n--- Examples of Incorrect 'Yes' Predictions (True Label was Fake/1) ---")
            for i, row in enumerate(error_cases.head(num_errors_to_print).itertuples()):
                print(f"  Example {i+1} (ID: {row.id}):")
                print(f"    Text: {row.text}")
                print(f"    True Label: {getattr(row, true_label_col)}")
                print(f"    Generated Text (Predicted Yes/0): {getattr(row, generated_text_col)}")
        else:
            print("\nNo incorrect 'Yes' predictions found (where True Label was Fake/1).")

if __name__ == '__main__':
    csv_file_path = 'results/blip/mini_experiment_blip_two_step_refined/all_model_outputs.csv'
    analyze_blip_outputs(csv_file_path, print_errors=True, num_errors_to_print=10) # Print up to 10 error examples 