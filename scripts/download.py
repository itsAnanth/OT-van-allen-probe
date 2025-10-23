# download data files
import os
import requests
import argparse
from tqdm import tqdm
import time
from multiprocessing import Pool


BASE_URL = "https://zenodo.org/records/6299967/files"
DATA_DIR = "dataset/raw"
MAX_SHARDS = 4
os.makedirs(DATA_DIR, exist_ok=True)
FILENAMES = ['ch3_data.7z', 'ch11_data.7z', 'ch14_data.7z', 'ch16_data.7z']
index_to_filename = lambda index: FILENAMES[index]

def download_single_file(index):
    
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Skipping {filepath} already exists")
        return True
    
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename} from {url}")
    
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 1024  # 1 MB
            
            temp_path = filepath + f".tmp"
            
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        
            os.rename(temp_path, filepath)
            print(f"Succesfully downloaded {filename}")
            return True
        
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            print("Cleaning up...")
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
                    
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} before retry ...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False
                    
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Van Allen probe data")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()
    
    ids_to_download = list(range(MAX_SHARDS))
    print(f"Downloading {len(ids_to_download)} data shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(download_single_file, ids_to_download), total=len(ids_to_download)))
        # results = pool.map(download_single_file, ids_to_download)
        
        
    successful = sum(1 for success in results if success)
    print(results)
    print(f"Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
        
        

