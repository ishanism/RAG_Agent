import os
from logger import debug_logger, info_logger, warning_logger, error_logger

def create_file(filename):
    debug_logger.debug(f"Attempting to create file: {filename}")
    
    if os.path.exists(filename):
        warning_logger.warning(f"File already exists: {filename}")
        return False
        
    try:
        with open(filename, 'w') as f:
            f.write('This is a test file.')
        info_logger.info(f"Successfully created file: {filename}")
        return True
    except Exception as e:
        error_logger.error(f"Failed to create file {filename}: {str(e)}")
        return False

if __name__ == '__main__':
    create_file('test_file.txt')