from classifier.datasets import multi_modal_dataset
import argparse
import torchxrayvision as xrv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_images", type=str, help="Directory of dataset")
    parser.add_argument("--path_csv", type=str, help="Path to dataset csv")
    return parser.parse_args()

nih_dataset = xrv.datasets.NIH_Dataset(
    imgpath="/path/to/images/",     
    csvpath="/path/to/Data_Entry_2017.csv",
    transform=None,                 
)


if __name__ == "__main__":
    args = parse_args()
    