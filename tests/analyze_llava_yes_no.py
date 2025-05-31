import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import argparse
import os

def extract_llava_prediction(generated_text):
    """
    Extracts a binary prediction from LLaVA's generated text.
    Assumes LLaVA's answer directly addresses if the post is misleading.
    
    - "Yes, this post is misleading..." or similar -> 1 (Fake)
    - "No, this post is not misleading..." or similar -> 0 (Real)
    Returns None if the pattern is not clearly matched.
    """
    if not isinstance(generated_text, str):
        return None
    
    # Find the start of the assistant's answer
    assistant_marker = "ASSISTANT:"
    marker_pos = generated_text.find(assistant_marker)
    
    if marker_pos == -1:
        # If ASSISTANT: marker is not found, try to process from beginning, 
        # but this might be less reliable for full prompt carry-overs.
        text_to_analyze = generated_text.lower().strip()
    else:
        text_to_analyze = generated_text[marker_pos + len(assistant_marker):].lower().strip()

    if not text_to_analyze: # Empty string after marker
        return None

    # More specific patterns for LLaVA
    if text_to_analyze.startswith("yes, this post is misleading") or \
       text_to_analyze.startswith("yes, the post is misleading") or \
       text_to_analyze.startswith("yes, it is misleading") or \
       text_to_analyze.startswith("yes. this post is misleading") or \
       text_to_analyze.startswith("yes. the post is misleading") or \
       text_to_analyze.startswith("yes. it is misleading") or \
       text_to_analyze.startswith("misleading."): # Simplified direct answer
        return 1  # Misleading -> Fake
    elif text_to_analyze.startswith("no, this post is not misleading") or \
         text_to_analyze.startswith("no, the post is not misleading") or \
         text_to_analyze.startswith("no, it is not misleading") or \
         text_to_analyze.startswith("no. this post is not misleading") or \
         text_to_analyze.startswith("no. the post is not misleading") or \
         text_to_analyze.startswith("no. it is not misleading") or \
         text_to_analyze.startswith("not misleading."): # Simplified direct answer
        return 0  # Not misleading -> Real
    
    # Broader checks if the above fail - can be made more robust
    # Check if "is misleading" appears early in the ASSISTANT's response part
    # Ensure not to take it from the USER's question part if marker was not found initially
    if "is misleading" in text_to_analyze and "is not misleading" not in text_to_analyze:
        # Check if 'is misleading' is within the first few words of the actual answer
        # This is a heuristic to avoid matching the phrase if it's deep in a long explanation without a clear yes/no
        # For instance, check the first 30 characters of text_to_analyze for the phrase
        if text_to_analyze.find("is misleading") < 30:
            return 1 
    if "is not misleading" in text_to_analyze:
        if text_to_analyze.find("is not misleading") < 30:
            return 0
    
    return None

def analyze_llava_outputs(csv_path, true_label_col='true_label', generated_text_col='generated_text', print_errors=True, num_errors_to_print=10):
    """
    Analyzes LLaVA's outputs:
    1. Reads the CSV.
    2. Extracts Yes/No predictions from 'generated_text'.
    3. Calculates and prints accuracy and classification report.
    4. Prints examples of misclassifications.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    if generated_text_col not in df.columns or true_label_col not in df.columns:
        print(f"Error: Required columns ('{generated_text_col}', '{true_label_col}') not in CSV.")
        return

    df['predicted_label_llava'] = df[generated_text_col].apply(extract_llava_prediction)
    
    # Filter out rows where prediction could not be extracted
    analyzable_df = df.dropna(subset=['predicted_label_llava'])
    if len(analyzable_df) == 0:
        print("Could not extract any Yes/No predictions from the LLaVA outputs after attempting to locate 'ASSISTANT:' marker.")
        if not df.empty and generated_text_col in df.columns:
            print(f"Output sample for manual check (ID: {df.iloc[0].get('id', 'N/A')}): {df.iloc[0][generated_text_col]}")
        return
    
    y_true = analyzable_df[true_label_col].astype(int)
    y_pred = analyzable_df['predicted_label_llava'].astype(int)

    print(f"--- LLaVA Prediction Analysis (Total samples evaluated: {len(analyzable_df)} out of {len(df)}) ---")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'], zero_division=0))
    
    counts = analyzable_df['predicted_label_llava'].value_counts(dropna=False)
    pred_not_misleading_0 = counts.get(0, 0)
    pred_misleading_1 = counts.get(1, 0)
    # Correctly calculate not_extracted based on original df length and analyzable_df length
    not_extracted = len(df) - len(analyzable_df)

    print("\nPrediction Summary:")
    print(f"  - Predicted 'Not Misleading' (mapped to 0): {pred_not_misleading_0}")
    print(f"  - Predicted 'Misleading' (mapped to 1): {pred_misleading_1}")
    print(f"  - Could not extract prediction: {not_extracted}")

    if print_errors:
        # Incorrect "Not Misleading" (Predicted 0, True 1 - False Negative)
        fn_errors = analyzable_df[(y_pred == 0) & (y_true == 1)]
        if not fn_errors.empty:
            print(f"\n--- Examples of Incorrect 'Not Misleading' Predictions (True Label was Fake/1) ---")
            for i, (idx, row) in enumerate(fn_errors.head(num_errors_to_print).iterrows()):
                print(f"  Example {i+1} (ID: {row['id']}):")
                print(f"    Text: {row.get('text', 'N/A')}")
                print(f"    True Label: {row[true_label_col]}")
                print(f"    Generated Text (Predicted Not Misleading/0): {row[generated_text_col]}")
        
        # Incorrect "Misleading" (Predicted 1, True 0 - False Positive)
        fp_errors = analyzable_df[(y_pred == 1) & (y_true == 0)]
        if not fp_errors.empty:
            print(f"\n--- Examples of Incorrect 'Misleading' Predictions (True Label was Real/0) ---")
            for i, (idx, row) in enumerate(fp_errors.head(num_errors_to_print).iterrows()):
                print(f"  Example {i+1} (ID: {row['id']}):")
                print(f"    Text: {row.get('text', 'N/A')}")
                print(f"    True Label: {row[true_label_col]}")
                print(f"    Generated Text (Predicted Misleading/1): {row[generated_text_col]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze LLaVA model outputs for Yes/No (Misleading/Not Misleading) predictions.")
    parser.add_argument("--csv_path", type=str, 
                        default="results/llava/cpu_runs/initial_llava_test_cpu/all_model_outputs.csv", 
                        help="Path to the LLaVA model output CSV file.")
    parser.add_argument("--true_label_col", type=str, default="true_label", help="Name of the column with true labels.")
    parser.add_argument("--generated_text_col", type=str, default="generated_text", help="Name of the column with LLaVA's generated text.")
    parser.add_argument("--no_print_errors", action="store_false", dest="print_errors", help="Suppress printing of error examples.")
    parser.add_argument("--num_errors", type=int, default=10, help="Number of error examples to print for each category.")

    args = parser.parse_args()
    
    analyze_llava_outputs(
        csv_path=args.csv_path,
        true_label_col=args.true_label_col,
        generated_text_col=args.generated_text_col,
        print_errors=args.print_errors,
        num_errors_to_print=args.num_errors
    ) 