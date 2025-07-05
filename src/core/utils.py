import json
import logging
import os
import re

# Setup logger for this module - use the setup_logger function itself for consistency
# This allows configuration if this module is run directly, or uses a root logger if imported.
logger = logging.getLogger(__name__) # Get a logger for this module
if not logger.handlers: # Configure only if no handlers are already attached (e.g. by a root config)
    # Basic configuration for direct use or if no other config is present
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Logging Setup ---
def setup_logger(name='mllm_project', level=logging.INFO, log_file=None, file_mode='a'):
    """
    Sets up a custom logger with unified log directory.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str, optional): Specific log file name. If None, uses 'mllm.log'.
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

        # File handler - always use unified log directory
        if log_file is None:
            log_file = 'mllm.log'
        
        # Determine project root and create unified log path
        # Try to find the project root by looking for README.md or .git
        current_dir = os.getcwd()
        project_root = current_dir
        
        # Look for project root indicators
        while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
            if os.path.exists(os.path.join(current_dir, 'README.md')) or \
               os.path.exists(os.path.join(current_dir, '.git')):
                project_root = current_dir
                break
            current_dir = os.path.dirname(current_dir)
        
        # Create unified log directory at project root
        unified_log_dir = os.path.join(project_root, 'logs')
        unified_log_file = os.path.join(unified_log_dir, log_file)
        
        # Create directory for log file if it doesn't exist
        if not os.path.exists(unified_log_dir):
            os.makedirs(unified_log_dir, exist_ok=True)
        
        fh = logging.FileHandler(unified_log_file, mode=file_mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Log the unified log location for debugging
        logger.debug(f"Logging to unified location: {unified_log_file}")
            
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
        logger.info(f"Data successfully saved to {file_path}")
    except IOError as e:
        logger.error(f"Error saving data to {file_path}: {e}")
    except TypeError as e:
        logger.error(f"Error serializing data to JSON: {e}")

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
        logger.error(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except IOError as e:
        logger.error(f"Error reading data from {file_path}: {e}")
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
