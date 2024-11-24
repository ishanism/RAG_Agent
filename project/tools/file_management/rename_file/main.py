
import os

def rename_file(old_name, new_name):
    if os.path.exists(old_name):
        os.rename(old_name, new_name)

if __name__ == '__main__':
    rename_file('test_file.txt', 'renamed_file.txt')