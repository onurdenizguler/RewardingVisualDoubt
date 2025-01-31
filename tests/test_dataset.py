import logging
import os

import dataset

logging.basicConfig(level=logging.INFO)

MIMIC_CXR_2_0_OFFICIAL_STUDY_COUNT = 227835
MIMIC_CXR_2_0_OFFICIAL_IMAGE_COUNT = 377110
MIMIC_CXR_DATASET_ROOT_DIR = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/files"


def test_create_mimic_cxr_dataset_df():
    logging.info("Testing create_mimic_cxr_dataset_df")
    logging.info("Loading the dataset_df")
    mimic_cxr_df = dataset.create_mimic_cxr_dataset_df()
    study_count = mimic_cxr_df["study_id"].nunique()
    image_count = len(mimic_cxr_df)
    print(
        f"Study count: {study_count} (expected: {MIMIC_CXR_2_0_OFFICIAL_STUDY_COUNT}) Offset: {MIMIC_CXR_2_0_OFFICIAL_STUDY_COUNT - study_count}"
    )
    print(
        f"Image count: {image_count} (expected: {MIMIC_CXR_2_0_OFFICIAL_IMAGE_COUNT}) Offset: {MIMIC_CXR_2_0_OFFICIAL_IMAGE_COUNT - image_count}"
    )

    # Sample randomly from image paths and assert that files exist
    NUM_IMG_PATHS_TO_SAMPLE = 100
    sampled_relative_img_paths = sampled_files = mimic_cxr_df["img_path"].sample(
        n=NUM_IMG_PATHS_TO_SAMPLE
    )
    for dir in sampled_relative_img_paths:
        if not os.path.exists(os.path.join(MIMIC_CXR_DATASET_ROOT_DIR, dir)):
            raise AssertionError(f"The file '{dir}' does not exist in the dataset directory.")


if __name__ == "__main__":
    test_create_mimic_cxr_dataset_df()
