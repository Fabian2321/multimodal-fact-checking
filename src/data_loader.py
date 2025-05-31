import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import requests # For downloading images
from io import BytesIO # For handling image bytes

# Default image transformations - can be customized
DEFAULT_IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to a common input size for many models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])

class FakedditDataset(Dataset):
    """
    Custom PyTorch Dataset for Fakeddit.
    Provides image data (either as transformed tensors or raw PIL images) and raw text strings.
    """
    def __init__(self, metadata_dir, metadata_file_name, 
                 downloaded_image_dir="data/downloaded_images",
                 transform="default_tensor", # Options: "default_tensor", None (for PIL), or a custom callable
                 image_id_col='id',
                 text_col='clean_title', label_col='2_way_label', image_url_col='image_url',
                 has_image_col='hasImage'):
        """
        Args:
            transform (str or callable, optional):
                - "default_tensor": Applies DEFAULT_IMAGE_TRANSFORMS.
                - None: Returns raw PIL.Image.Image objects.
                - callable: A custom transform to be applied to the PIL image.
        """
        self.metadata_dir = metadata_dir
        self.metadata_file_name = metadata_file_name
        self.downloaded_image_dir = downloaded_image_dir
        os.makedirs(self.downloaded_image_dir, exist_ok=True)

        self.image_id_col = image_id_col
        self.text_col = text_col
        self.label_col = label_col
        self.image_url_col = image_url_col
        self.has_image_col = has_image_col

        metadata_path = os.path.join(self.metadata_dir, self.metadata_file_name)
        self.metadata = pd.DataFrame() # Initialize as empty DataFrame

        try:
            # Modified pd.read_csv for robustness and explicit engine
            temp_metadata = pd.read_csv(metadata_path, sep='\\t', engine='python', on_bad_lines='warn')
            print(f"INFO: Successfully read {metadata_path} using python engine. Loaded columns: {temp_metadata.columns.tolist()}")
            print(f"INFO: Initial rows in metadata: {len(temp_metadata)}")

            required_cols = [self.has_image_col, self.image_id_col, self.image_url_col, self.text_col, self.label_col]
            if all(col in temp_metadata.columns for col in required_cols):
                print(f"INFO: All required columns found: {required_cols}")

                # Convert has_image_col to boolean if it's an object type
                if temp_metadata[self.has_image_col].dtype == 'object':
                    print(f"INFO: Converting '{self.has_image_col}' to boolean.")
                    temp_metadata[self.has_image_col] = temp_metadata[self.has_image_col].apply(
                        lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x)
                    )
                
                self.metadata = temp_metadata[temp_metadata[self.has_image_col] == True].copy()
                print(f"INFO: Rows after filtering by '{self.has_image_col}' == True: {len(self.metadata)}")
                
                self.metadata.dropna(subset=required_cols, inplace=True)
                print(f"INFO: Rows after dropping NaNs in required columns: {len(self.metadata)}")
                
                # Filter for valid image URLs
                self.metadata = self.metadata[self.metadata[self.image_url_col].str.startswith('http', na=False)].copy()
                print(f"INFO: Rows after filtering for valid image URLs (startswith 'http'): {len(self.metadata)}")
            else:
                missing = [col for col in required_cols if col not in temp_metadata.columns]
                print(f"WARNING: Metadata file {metadata_path} missing columns: {missing}")
            if self.metadata.empty:
                print(f"WARNING: Metadata empty after filtering in {metadata_path}.")

        except FileNotFoundError:
            print(f"ERROR: Metadata file not found: {metadata_path}")
        except Exception as e:
            print(f"Error loading metadata {metadata_path}: {e}")

        if transform == "default_tensor":
            self.transform_to_apply = DEFAULT_IMAGE_TRANSFORMS
        elif callable(transform):
            self.transform_to_apply = transform
        else: # Includes transform == None
            self.transform_to_apply = None

    def __len__(self):
        return len(self.metadata)

    def get_image_path(self, image_id):
        """Constructs the local path for a given image ID."""
        # Try to determine extension from metadata if possible, or default
        # This logic might need to be more robust if image_url isn't always available here
        # or if IDs are not unique enough without extension context.
        file_extension = ".jpg" # Default
        try:
            # Attempt to find the original URL to infer extension; this is a bit indirect
            # A better way might be to store the determined extension during _download_image or ensure IDs are unique file stems
            row = self.metadata[self.metadata[self.image_id_col] == image_id].iloc[0]
            url = row.get(self.image_url_col, "")
            if isinstance(url, str):
                if url.lower().endswith(".png"): file_extension = ".png"
                elif url.lower().endswith(".jpeg"): file_extension = ".jpeg"
        except (IndexError, KeyError):
            # logger might be useful here if passed or a global one used
            print(f"Warning: Could not determine original extension for image_id {image_id} in get_image_path. Defaulting to .jpg")
            pass

        sanitized_image_id = str(image_id).replace('/', '_').replace('\\\\', '_')
        return os.path.join(self.downloaded_image_dir, f"{sanitized_image_id}{file_extension}")

    def _download_image(self, image_id, url):
        # Generate a file extension based on typical web image formats, default to .jpg
        file_extension = ".jpg" 
        if isinstance(url, str):
            if url.lower().endswith(".png"):
                file_extension = ".png"
            elif url.lower().endswith(".jpeg"):
                file_extension = ".jpeg"
        else: # If URL is not a string (e.g. float if column had mixed types before dropna)
            print(f"Warning: Invalid URL type for image id {image_id}: {url}. Skipping download.")
            return None
        
        sanitized_image_id = str(image_id).replace('/', '_').replace('\\\\', '_')
        local_img_path = os.path.join(self.downloaded_image_dir, f"{sanitized_image_id}{file_extension}")

        if os.path.exists(local_img_path):
            try:
                image = Image.open(local_img_path).convert('RGB')
                return image
            except Exception as e:
                print(f"Warning: Could not open existing image {local_img_path}: {e}. Will attempt re-download.")
                try:
                    os.remove(local_img_path)
                except OSError:
                    pass

        try:
            # print(f"Downloading image for id {image_id} from {url} to {local_img_path}") # Verbose
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            response.raise_for_status()
            
            # Ensure content is not too small (very basic check for empty/error pages)
            if 'content-length' in response.headers and int(response.headers['content-length']) < 1024:
                print(f"Warning: Content length for {url} (id {image_id}) is very small. Potential issue.")

            image = Image.open(BytesIO(response.content)).convert('RGB')
            image.save(local_img_path)
            return image
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not download image for id {image_id} from {url}. Error: {e}")
        except IOError as e:
            print(f"Warning: Downloaded content for id {image_id} from {url} is not a valid image or cannot be opened. Error: {e}")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while downloading/saving image for id {image_id} from {url}. Error: {e}")
        return None

    def __getitem__(self, idx):
        if self.metadata.empty:
            raise IndexError("Dataset metadata is empty.") 
            
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            row = self.metadata.iloc[idx]
            img_id = row[self.image_id_col]
            image_url = row[self.image_url_col]
            text_content = str(row[self.text_col]) 
            label = row[self.label_col]

            pil_image = self._download_image(img_id, image_url)
            
            output_image = None
            if pil_image is None: 
                print(f"Warning: Image for id {img_id} failed to load. Creating dummy PIL.")
                pil_image = Image.new('RGB', (224, 224), (200, 200, 200)) # Grey placeholder PIL
            
            if self.transform_to_apply:
                output_image = self.transform_to_apply(pil_image)
            else: # Return raw PIL image
                output_image = pil_image

            try:
                numeric_label = int(label)
            except ValueError:
                print(f"Warning: Label '{label}' for id {img_id} not int. Using -1.")
                numeric_label = -1 

            sample = {
                'id': str(img_id),
                'image': output_image, 
                'text': text_content,
                'label': numeric_label, 
            }

        except KeyError as e:
            print(f"KeyError: {e} for index {idx}. Check column names.")
            return None 
        except Exception as e:
            print(f"Error at index {idx}, ID {self.metadata.iloc[idx].get(self.image_id_col, 'UNKNOWN')}: {e}")
            return None
        return sample

# Example usage (for testing the FakedditDataset class)
if __name__ == '__main__':
    print("--- Testing FakedditDataset --- ")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    METADATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
    DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, 'data', 'downloaded_test_images')
    METADATA_FILE = 'multimodal_train.csv'

    print(f"Project root assumed: {PROJECT_ROOT}")
    print(f"Metadata: {os.path.join(METADATA_DIR, METADATA_FILE)}")
    print(f"Download dir: {DOWNLOAD_DIR}")

    print("\n--- Test 1: Loading with default tensor transform ---")
    dataset_tensor = FakedditDataset(METADATA_DIR, METADATA_FILE, DOWNLOAD_DIR)
    if len(dataset_tensor) > 0:
        sample_tensor = dataset_tensor[0]
        if sample_tensor:
            print(f"Sample 0 (tensor): ID {sample_tensor['id']}, Image type: {type(sample_tensor['image'])}, Image shape/val: {sample_tensor['image'].shape if isinstance(sample_tensor['image'], torch.Tensor) else 'Not a Tensor'}, Text: \"{sample_tensor['text'][:30]}...\", Label: {sample_tensor['label']}")
    else:
        print("Tensor dataset is empty.")

    print("\n--- Test 2: Loading with transform=None (to get PIL images) ---")
    dataset_pil = FakedditDataset(METADATA_DIR, METADATA_FILE, DOWNLOAD_DIR, transform=None)
    if len(dataset_pil) > 0:
        sample_pil = dataset_pil[0]
        if sample_pil:
            print(f"Sample 0 (PIL): ID {sample_pil['id']}, Image type: {type(sample_pil['image'])}, Image size: {sample_pil['image'].size if isinstance(sample_pil['image'], Image.Image) else 'Not a PIL Image'}, Text: \"{sample_pil['text'][:30]}...\", Label: {sample_pil['label']}")
    else:
        print("PIL dataset is empty.")

    print("\n--- Test 3: DataLoader with PIL images (requires collate_fn that handles PIL) ---")
    if len(dataset_pil) > 0:
        from torch.utils.data import DataLoader
        
        def collate_pil_batch(batch):
            # Custom collate for batches of samples where 'image' is a PIL Image
            # This will be processed by model-specific processor later
            ids = [item['id'] for item in batch if item]
            images = [item['image'] for item in batch if item]
            texts = [item['text'] for item in batch if item]
            labels = torch.tensor([item['label'] for item in batch if item], dtype=torch.long)
            if not ids: return None
            return {'id': ids, 'image': images, 'text': texts, 'label': labels}

        dataloader_pil = DataLoader(dataset_pil, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_pil_batch)
        try:
            pil_batch = next(iter(dataloader_pil))
            if pil_batch:
                print(f"PIL Batch: Image type in batch: {type(pil_batch['image'][0]) if pil_batch['image'] else 'N/A'}, Num images: {len(pil_batch['image']) if pil_batch['image'] else 0}")
        except StopIteration:
            print("PIL DataLoader is empty.")
        except Exception as e:
            print(f"Error iterating PIL DataLoader: {e}")
    else:
        print("PIL dataset empty, skipping DataLoader test.") 