import json
import logging
import os
import re

# --- Logging Setup ---
def setup_logger(name='mllm_project', level=logging.INFO, log_file=None, file_mode='a'):
    """
    Sets up a custom logger.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str, optional): Path to a file to save logs. If None, logs to console only.
        file_mode (str): Mode to open log file ('a' for append, 'w' for write).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already has them
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (optional)
        if log_file:
            # Create directory for log file if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            fh = logging.FileHandler(log_file, mode=file_mode)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger

# --- JSON Read/Write Utilities ---
def save_to_json(data, file_path, indent=4):
    """
    Saves a Python dictionary or list to a JSON file.

    Args:
        data: The Python object (dict, list) to save.
        file_path (str): The path to the JSON file.
        indent (int): Indentation level for pretty printing.
    """
    try:
        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        # print(f"Data successfully saved to {file_path}") # Optional: for verbosity
    except IOError as e:
        print(f"Error saving data to {file_path}: {e}")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")

def load_from_json(file_path):
    """
    Loads data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The loaded Python object, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return None
    except IOError as e:
        print(f"Error reading data from {file_path}: {e}")
        return None

# --- Text Normalization (Placeholder) ---
def normalize_text_basic(text):
    """
    Basic text normalization placeholder.
    - Converts to lowercase.
    - Removes extra whitespace.
    (This is very basic; real normalization might involve more steps like
     punctuation removal/handling, unicode normalization, etc., depending on needs
     and whether the model's tokenizer already handles these.)
    
    Args:
        text (str): The input text.

    Returns:
        str: The normalized text.
    """
    if not isinstance(text, str):
        return "" # Or raise an error, or convert, depending on desired behavior
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single, and strip
    return text

# --- Example Usage (for testing utils.py directly) ---
if __name__ == '__main__':
    # Test logger
    logger = setup_logger('utils_test', level=logging.DEBUG, log_file='temp_test_logs/utils_test.log')
    logger.debug("This is a debug message from utils_test.")
    logger.info("This is an info message from utils_test.")
    logger.warning("This is a warning message.")
    print(f"Check 'temp_test_logs/utils_test.log' for file logging output.")

    # Test JSON utilities
    test_data_dict = {'name': 'Test Project', 'version': 1.0, 'tags': ['test', 'multimodal']}
    test_data_list = [{'id': 1, 'text': 'Example 1'}, {'id': 2, 'text': 'Example 2 日本語'}]
    json_dict_path = 'temp_test_data/test_dict.json'
    json_list_path = 'temp_test_data/test_list.json'

    save_to_json(test_data_dict, json_dict_path)
    loaded_dict = load_from_json(json_dict_path)
    print(f"Loaded dictionary from JSON: {loaded_dict}")

    save_to_json(test_data_list, json_list_path)
    loaded_list = load_from_json(json_list_path)
    print(f"Loaded list from JSON: {loaded_list}")
    
    # Test text normalization
    sample_text = "  This is an Example TEXT with   Extra Spaces!!  "
    normalized = normalize_text_basic(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Normalized text: '{normalized}'")
    
    sample_non_string = 123
    normalized_non_string = normalize_text_basic(sample_non_string)
    print(f"Original non-string: {sample_non_string}, Normalized: '{normalized_non_string}'")


    # Clean up test files/dirs
    if os.path.exists(json_dict_path): os.remove(json_dict_path)
    if os.path.exists(json_list_path): os.remove(json_list_path)
    if os.path.exists('temp_test_data'): os.rmdir('temp_test_data')
    if os.path.exists('temp_test_logs/utils_test.log'): os.remove('temp_test_logs/utils_test.log')
    if os.path.exists('temp_test_logs'): os.rmdir('temp_test_logs')
    print("Test cleanup finished.")
