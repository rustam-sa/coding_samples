"""
Utility module for creating and managing directories within a file system.

This module provides a class that simplifies the process of creating and managing directories,
allowing for the organization of files and other directories within a specified main directory.
"""

from pathlib import Path

class DirectoryManager:
    """A class to manage directories within a specified main directory.
    
    Attributes:
        main_directory (Path): The main directory within which new directories will be managed.
    
    Methods:
        create_main_directory(): Creates the main directory if it doesn't exist.
        create_new_dir(new_dir_name): Creates a new subdirectory within the main directory.
    """

    def __init__(self, main_directory):
        """Initialize the DirectoryManager with a specified main directory.
        
        Args:
            main_directory (str): The path to the main directory to be managed.
        """
        self.main_directory = Path(main_directory)
        self.create_main_directory()
        
    def create_main_directory(self):
        """Create the main directory if it doesn't already exist.
        
        This method checks if the main directory exists, and if not, creates it.
        Outputs a message indicating whether the directory was created or already exists.
        """
        new_dir_name = Path(self.main_directory)
        if not new_dir_name.exists():
            new_dir_name.mkdir()
            print(f"Directory '{str(new_dir_name)}' created successfully!")
        else:
            print(f"Directory '{str(new_dir_name)}' already exists.")
        
    def create_new_dir(self, new_dir_name):
        """Create a new subdirectory within the main directory if it doesn't already exist.
        
        Args:
            new_dir_name (str): The name of the new subdirectory to create.
        
        Returns:
            Path: The path of the newly created subdirectory, or of the existing one if it was already present.
            
        Outputs a message indicating whether the directory was created or already exists.
        """
        new_dir_path = Path(self.main_directory, new_dir_name)
        if not new_dir_path.exists():
            new_dir_path.mkdir()
            print(f"Directory '{str(new_dir_path)}' created successfully!")
        else:
            print(f"Directory '{str(new_dir_path)}' already exists.")
        
        return new_dir_path