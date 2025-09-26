import os
from bing_image_downloader import downloader

# auto data from bing (this is so shit but idk why other shit did not work for me bruh)

def save_images(query, num_images=10):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_parent = os.path.join(script_dir, 'data')
    os.makedirs(data_parent, exist_ok=True)

    print(f"Downloading images for: {query} ...")
    downloader.download(
        query,
        limit=num_images,
        output_dir=data_parent,
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True
    )

    folder_name = query.replace(' ', '_')
    data_folder = os.path.join(data_parent, folder_name)
    print(f"Downloaded images to {data_folder}")

if __name__ == "__main__":
    search_query = input("Enter what you want to train the AI with: ").strip()
    count = int(input("Count: "))
    save_images(search_query, num_images=count)
