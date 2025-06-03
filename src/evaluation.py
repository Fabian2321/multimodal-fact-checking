import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging # Import logging

# Setup logger for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def calculate_metrics(y_true, y_pred, y_prob=None, average='binary'):
    """
    Calculates and returns a dictionary of common classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Predicted probabilities for the positive class.
                                       Required for AUC if that's added.
        average (str, optional): Averaging method for precision, recall, F1-score.
                                 Options: 'binary', 'micro', 'macro', 'weighted', None.
                                 Default is 'binary'.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, f1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # TODO: Add ROC AUC calculation if y_prob is provided and task is binary.
    # from sklearn.metrics import roc_auc_score
    # if y_prob is not None and average == 'binary':
    #     try:
    #         metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    #     except ValueError as e:
    #         logger.warning(f"Could not calculate ROC AUC: {e}. Ensure y_true contains both classes for binary classification.")
    #         metrics['roc_auc'] = None
            
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title='Confusion Matrix'):
    """
    Plots and optionally saves a confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list of str): Names of the classes for labels.
        save_path (str, optional): Path to save the plot. If None, plot is shown.
        title (str, optional): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def evaluate_model_outputs(results_df, true_label_col='true_labels', pred_label_col='predicted_labels', 
                           generated_text_col='generated_text', # New argument for generated text
                           report_path=None, figures_dir=None):
    """
    Main function to evaluate model outputs from a DataFrame.
    Calculates metrics, generates a confusion matrix, and optionally saves a report.
    Also includes generated text in the report if available.

    Args:
        results_df (pd.DataFrame): DataFrame containing true labels and predicted labels.
        true_label_col (str): Column name for true labels.
        pred_label_col (str): Column name for predicted labels.
        generated_text_col (str): Column name for generated text/explanations (optional).
        report_path (str, optional): Path to save the evaluation report (e.g., 'results/reports/clip_evaluation.txt').
        figures_dir (str, optional): Directory to save figures like the confusion matrix (e.g., 'results/figures/').
    """
    if results_df.empty:
        logger.warning("Results DataFrame is empty. Nothing to evaluate.")
        return None

    y_true = results_df[true_label_col]
    y_pred = results_df[pred_label_col]
    
    # Assuming binary classification for Fakeddit (e.g., 0: Real, 1: Fake)
    # Adapt class_names if your labels are different.
    class_names = ['Real', 'Fake'] 
    
    logger.info("\n--- Overall Metrics ---")
    metrics = calculate_metrics(y_true, y_pred, average='binary') # Or 'macro'/'weighted' for multi-class
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.capitalize()}: {value:.4f}")

    if figures_dir:
        cm_save_path = os.path.join(figures_dir, "confusion_matrix.png")
    else:
        cm_save_path = None
        
    logger.info("\n--- Confusion Matrix ---")
    plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=cm_save_path)

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write("Evaluation Report\n")
            f.write("====================\n")
            # Overall metrics (macro average)
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro Precision: {macro_precision:.4f}\n")
            f.write(f"Macro Recall: {macro_recall:.4f}\n")
            f.write(f"Macro F1-score: {macro_f1:.4f}\n")
            f.write("\n\n--- Classification Report (Per Class) ---\n")
            
            # Detailed classification report
            from sklearn.metrics import classification_report
            report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
            f.write(report_str)
            
            if generated_text_col in results_df.columns and results_df[generated_text_col].notna().any():
                f.write("\n\n--- Sample Generated Texts/Explanations ---\n")
                # Write a few examples of generated text, true labels, and predicted labels
                sample_size = min(5, len(results_df))
                for i in range(sample_size):
                    f.write(f"  Sample {i+1} (ID: {results_df.iloc[i].get('id', 'N/A')}:\n")
                    f.write(f"    True: {results_df.iloc[i][true_label_col]}, Predicted: {results_df.iloc[i][pred_label_col]}\n")
                    f.write(f"    Generated Text: {results_df.iloc[i][generated_text_col]}\n")
            
            f.write("\n--- Placeholder for Advanced Explainability Metrics ---\n")
            f.write("Quality of generated explanations (human-annotated relevance): TODO\n")
            f.write("Visualization of attention weights or Grad-CAM heatmaps: TODO (qualitative analysis)\n")
            # TODO: Add more details to the report, e.g., per-class metrics, sample misclassifications.
        logger.info(f"Evaluation report saved to {report_path}")
        
    return metrics

if __name__ == '__main__':
    print("--- Testing Evaluation Functions ---")
    # Create dummy data for testing
    
    # Example: Binary classification
    y_true_binary = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    y_pred_binary = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    # y_prob_binary = np.array([0.1, 0.4, 0.2, 0.8, 0.3, 0.9, 0.4, 0.6, 0.1, 0.7]) # For ROC AUC
    
    print("\nTesting Binary Classification Metrics:")
    binary_metrics = calculate_metrics(y_true_binary, y_pred_binary, average='binary')
    for k, v in binary_metrics.items():
        print(f"{k}: {v}")
        
    print("\nTesting Plot Confusion Matrix (Binary):")
    # Create a dummy figures directory for testing save
    test_figures_dir = "temp_test_figures"
    os.makedirs(test_figures_dir, exist_ok=True)
    cm_test_save_path = os.path.join(test_figures_dir, "test_cm_binary.png")

    plot_confusion_matrix(y_true_binary, y_pred_binary, class_names=['Class 0', 'Class 1'], save_path=cm_test_save_path)
    if os.path.exists(cm_test_save_path):
        print(f"Test confusion matrix saved to {cm_test_save_path}")
    
    # Example: DataFrame for evaluate_model_outputs
    import pandas as pd
    results_data = {
        'id': [f'id_{i}' for i in range(len(y_true_binary))],
        'true_labels': y_true_binary,
        'predicted_labels': y_pred_binary
    }
    test_df = pd.DataFrame(results_data)
    
    print("\nTesting evaluate_model_outputs:")
    test_report_path = "temp_test_reports/evaluation_summary.txt"
    # Add dummy generated text for testing
    test_df['generated_text'] = [f"Explanation for sample {i}" if i % 2 == 0 else None for i in range(len(y_true_binary))]

    evaluate_model_outputs(test_df, report_path=test_report_path, figures_dir=test_figures_dir, generated_text_col='generated_text')

    # Clean up dummy directories/files
    if os.path.exists(cm_test_save_path):
        os.remove(cm_test_save_path)
    if os.path.exists(test_report_path):
        os.remove(test_report_path)
    if os.path.exists(os.path.dirname(test_report_path)):
        os.rmdir(os.path.dirname(test_report_path))
    if os.path.exists(test_figures_dir):
        os.rmdir(test_figures_dir)
        
    print("\n--- Evaluation script tests finished ---")
