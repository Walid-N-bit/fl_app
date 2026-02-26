import glob
from pathlib import Path
from wheat_data_utils import imgs_data_to_csv

dataset_path = "compressed_images_wheat"


# client 1
# train_folders = [
#     "880/Fusarium",
#     "880/Black germ",
#     "880/Sound",
#     "880/Spotted",
#     "880/Sprouted",
#     "880/4TH Batch/Broken",
#     "880/4TH Batch/Insect",
#     "880/4TH Batch/Moldy",
# ]
# test_folder = [
#     "test/880/Fusarium",
#     "test/880/Black germ",
#     "test/880/Sound",
#     "test/880/Spotted",
#     "test/880/Sprouted",
#     "test/880/First Batch/Broken",
#     "test/880/First Batch/Insect",
#     "test/880/First Batch/Moldy",
# ]
# imgs_data_to_csv(dataset_path, train_folders, test_folder)

# client 2
# train_folders = [
#     "864/Fusarium",
#     "864/Black germ",
#     "864/Sound",
#     "864/First Batch/Spotted",
#     "864/First Batch/Sprouted",
#     "864/First Batch/Broken",
#     "864/First Batch/Insect",
#     "864/First Batch/Moldy",
# ]
# test_folder = [
#     "test/864/Fusarium",
#     "test/864/Black germ",
#     "test/864/Sound",
#     "test/864/Fourth Batch//Spotted",
#     "test/864/Fourth Batch//Sprouted",
#     "test/864/Fourth Batch/Broken",
#     "test/864/Fourth Batch/Insect",
#     "test/864/Fourth Batch/Moldy",
# ]
# imgs_data_to_csv(dataset_path, train_folders, test_folder)
