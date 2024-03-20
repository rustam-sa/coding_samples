"""
utility for creating directories for simple organization
"""

from pathlib import Path


class DirectoryManager:
    def __init__(self, main_directory):
        self.main_directory = Path(main_directory)
        self.create_main_directory()
        
    
    def create_main_directory(self):
        # Creates the main directory if it doesn't already exist
        new_dir_name = Path(self.main_directory)
        if not new_dir_name.exists():
            new_dir_name.mkdir()
            print(f"Directory '{str(new_dir_name)}' created successfully!")
        else:
            print(f"Directory '{str(new_dir_name)}' already exists.")
        
    
    def create_new_dir(self, new_dir_name):
        # Create a directory if it doesn't already exist
        new_dir_name = Path(self.main_directory, new_dir_name)
        if not new_dir_name.exists():
            new_dir_name.mkdir()
            print(f"Directory '{str(new_dir_name)}' created successfully!")
        else:
            print(f"Directory '{str(new_dir_name)}' already exists.")
            
        return new_dir_name
    
        