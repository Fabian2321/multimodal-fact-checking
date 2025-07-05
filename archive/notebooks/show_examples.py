import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os

def show_example(df, index, save_dir='results/examples'):
    """
    Saves a visualization of an example from the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe containing the model outputs.
        index (int): The index of the example to show.
        save_dir (str): The directory to save the output image.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if index >= len(df):
        print(f"Error: Index {index} is out of bounds for dataframe with length {len(df)}.")
        return
    
    row = df.iloc[index]
    image_path = row['image_path']
    text = row['text']
    true_label = row['true_label']
    item_id = row['id']

    fig, ax = plt.subplots()

    # --- Display Image ---
    try:
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            # Path is now relative to project root, inside the image data folder
            full_image_path = os.path.join('data/downloaded_fakeddit_images', os.path.basename(image_path))
            img = Image.open(full_image_path)
        
        ax.imshow(img)
        
    except Exception as e:
        print(f"Could not load image at {image_path}: {e}")
        placeholder = Image.new('RGB', (200, 200), color = 'grey')
        ax.imshow(placeholder)
        ax.text(100, 100, 'Image not available', ha='center', va='center')

    ax.axis('off')
    
    # --- Add Text Details as Title ---
    title_str = f"ID: {item_id}\\nText: '{text}'\\n"
    title_str += f"--> True Label: {'Fake' if true_label == 1 else 'Real'} ({true_label})\\n"
    
    # Model-specific outputs
    if 'predicted_labels' in row:
        predicted_label = row['predicted_labels']
        title_str += f"--> CLIP Predicted: {'Fake' if predicted_label == 1 else 'Real'} ({predicted_label}) | "
    if 'scores' in row:
        title_str += f"Score: {row['scores']:.2f}\\n"
    
    if 'generated_text_decision' in row:
        title_str += f"--> BLIP/LLaVA Decision: {row['generated_text_decision']}\\n"
    if 'generated_text_explanation' in row:
        title_str += f"    Explanation: {row['generated_text_explanation']}"
        
    fig.suptitle(title_str, x=0.01, y=0.99, ha='left', va='top', fontsize=8, wrap=True)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # --- Save Figure ---
    save_path = os.path.join(save_dir, f"example_{item_id}_{index}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved example to {save_path}")
    plt.close()


def main():
    result_paths = {
        'clip_optimized': 'results/clip/clip_final_optimal_thresh/all_model_outputs.csv',
        'blip_two_step': 'results/blip/blip_test_prompt_optionB_s1t20_s2t50/all_model_outputs.csv',
    }
    
    # Load the dataframes
    dfs = {}
    for name, path in result_paths.items():
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
            print(f"Loaded {name} with {len(dfs[name])} rows.")
        else:
            print(f"Warning: Could not find file for {name} at {path}")

    # Generate and save a few examples
    if 'clip_optimized' in dfs:
        print("\\n--- Generating examples from CLIP results ---")
        for i in range(min(5, len(dfs['clip_optimized']))):
            show_example(dfs['clip_optimized'], i)
            
    if 'blip_two_step' in dfs:
        print("\\n--- Generating examples from BLIP results ---")
        for i in range(min(5, len(dfs['blip_two_step']))):
            show_example(dfs['blip_two_step'], i)

if __name__ == '__main__':
    main() 