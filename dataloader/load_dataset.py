from pathlib import Path

import opendatasets as od
import os
import tarfile


class LoadDataset:
    """
    Class to load the dataset from Kaggle and extract it.
    """

    def __init__(self, url='https://www.kaggle.com/datasets/atulanandjha/lfwpeople', save_path=None, extract_path=None):
        """
        Initialize the LoadDataset class.
        """
        self.url = url
        self.save_path = save_path or os.getcwd()
        self.dataset_name = self.url.split('/')[-1]
        self.dataset_path = os.path.join(self.save_path, self.dataset_name)
        self.extract_path = extract_path or os.path.join(self.dataset_path, "extracted")

        self._load()
        self.pairs = self._parse_lfw_pairs(
            pairs_txt=os.path.join(self.dataset_path, "pairs.txt"),
            images_dir=os.path.join(self.extract_path, "lfw_funneled")
        )

    def _download(self, force_download:bool=False):
        """
        Download the dataset from Kaggle.
        """
        if not os.path.exists(self.dataset_path):
            print("Downloading dataset...")
            od.download(self.url, data_dir=self.save_path)
        else:
            print("Dataset already downloaded.")

    def _load(self):
        """
        Load the dataset from Kaggle and extract it.
        """
        self._download()
        tgz_path = os.path.join(self.dataset_path, "lfw-funneled.tgz")

        if not os.path.exists(tgz_path):
            raise FileNotFoundError(f"Dataset file {tgz_path} not found. Please check the download path.")

        if not os.path.exists(self.extract_path):
            print("Extracting dataset...")
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=self.extract_path)
            print("Dataset extracted successfully.")
        else:
            print("Dataset already extracted.")

    def _parse_lfw_pairs(self, pairs_txt, images_dir):
        """
        Parses LFW pairs.txt for face verification.

        Returns:
          List of tuples: (img_path_1, img_path_2, label)
          where label = 1 for same, 0 for different.
        """
        pairs_txt_path = pairs_txt
        images_dir_path = Path(images_dir)
        pairs = []
        with open(pairs_txt_path, 'r') as f:
            num_folds, num_pairs = map(int, f.readline().split())

            for fold in range(num_folds):
                # Same-person pairs
                for _ in range(num_pairs):
                    name, i, j = f.readline().split()
                    img1 = images_dir_path / name / f"{name}_{int(i):04d}.jpg"
                    img2 = images_dir_path / name / f"{name}_{int(j):04d}.jpg"
                    pairs.append((img1, img2, 1))
                # Different-person pairs
                for _ in range(num_pairs):
                    name1, i, name2, j = f.readline().split()
                    img1 = images_dir_path / name1 / f"{name1}_{int(i):04d}.jpg"
                    img2 = images_dir_path / name2 / f"{name2}_{int(j):04d}.jpg"
                    pairs.append((img1, img2, 0))
        return pairs