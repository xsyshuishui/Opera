import argparse
import datasets
import copy
import os

def process_raw_image(image: dict):
    from PIL import Image
    from io import BytesIO
    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image

parser = argparse.ArgumentParser(description="Check parquet files and remove broken image entries")
parser.add_argument("--parquet-dir", required=True, help="Directory containing .parquet files")
parser.add_argument("--lq-dir",      required=True, help="Directory of LQ images (for deletion)")
args = parser.parse_args()

parquet_dir = args.parquet_dir
data_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
print(data_files)

x = []
for parquet_file in data_files:
    dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
    print(f"Testing {parquet_file}")
    for row_dict in dataframe:
        print(row_dict.get('prompt'))
        try:
            origin_images = [process_raw_image(image) for image in row_dict.get("images")]
        except Exception as e:
            print(e)
            print(f"IMG: {row_dict.get('extra_info')}")
            x.append(row_dict.get('extra_info').get('image_name'))
print(x)

# delete these images
for img_name in x:
    img_path = os.path.join(args.lq_dir, img_name)
    if os.path.exists(img_path):
        os.remove(img_path)
        print(f"Deleted {img_path}")
