import os
import argparse
import heapq

parser = argparse.ArgumentParser(description='get grid search results')
parser.add_argument('--dir', type=str, help='dataset')
args = parser.parse_args()

def find_lpips_files(root_dir, fname="lpips.txt"):
    """
    Recursively searches the directory for files named 'lpips.txt' and returns a list of tuples,
    each containing the number read from the file and the file's path.
    """
    lpips_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == fname:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        number = float(f.read().strip().split(' ')[1])
                        lpips_files.append((number, file_path))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return lpips_files

def top_five_lpips_files(root_dir, fname):
    """
    Finds the top 5 lpips.txt files with the smallest numbers.
    """
    lpips_files = find_lpips_files(root_dir, fname)
    # Using a heap to keep track of the smallest numbers
    if not lpips_files:
        return []

    # Get the top 3 smallest numbers using a min-heap
    if fname == "lpips.txt":
        smallest_files = heapq.nsmallest(5, lpips_files)
    else:
        smallest_files = heapq.nlargest(5, lpips_files)
    
    return [file for file in smallest_files]

# Example usage:
root_directory = args.dir
top_files = top_five_lpips_files(root_directory, "lpips.txt")
print("Top 5 lpips.txt files with the best numbers:")
for path in top_files:
    print(path)

top_files = top_five_lpips_files(root_directory, "psnr.txt")
print("Top 5 psnr.txt files with the best numbers:")
for path in top_files:
    print(path)

top_files = top_five_lpips_files(root_directory, "ms_ssim.txt")
print("Top 5 ms_ssim.txt files with the best numbers:")
for path in top_files:
    print(path)