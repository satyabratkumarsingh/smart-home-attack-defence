import os

def find_all_file_names():
  directory = './CICIoT2023/'
  all_file_paths = []
  for root, dirs, files in os.walk(directory):
    for filename in files:
        full_path = os.path.join(root, filename)
        all_file_paths.append(full_path)

  return all_file_paths

