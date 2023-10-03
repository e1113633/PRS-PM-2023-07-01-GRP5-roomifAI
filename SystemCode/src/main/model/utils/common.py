import os

def get_file_type(path):
    return os.path.splitext(path)[1].lower()