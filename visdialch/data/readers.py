"""
A Reader simply reads data from disk and returns it almost as is. Readers should be utilized by 
torch ``Dataset``s. Any type of data pre-processing is not recommended in the reader, such as
tokenizing words to integers, embedding tokens, or passing an image through a pre-trained CNN.

Each Reader should be initialized by one or more file paths, and should provide access to a
single data instance by ``image_id`` of VisDial images (implement ``__getitem__``).

Note: I should have made a base Reader class and let these two extend it, but this way they are
independent and fit to be copy-pasted in some other codebase. :) 
"""

import copy
import json
from typing import Dict, List, Union

import h5py
# a bit slow, and just splits sentences to list of words, can be doable in the reader
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class VisDialJsonReader(object):
    """
    A simple reader for VisDial v1.0 data. The json file must have the same structure as mentioned
    on ``https://visualdialog.org/data``.

    Parameters
    ----------
    visdial_jsonpath : str
        Path to a json file containing VisDial v1.0 train, val or test data.
    """

    def __init__(self, visdial_jsonpath: str):
        with open(visdial_jsonpath, "r") as visdial_file:
            visdial_data = json.load(visdial_file)
            self._split = visdial_data["split"]

            self.questions = visdial_data["data"]["questions"]
            self.answers = visdial_data["data"]["answers"]

            # add empty question, answer at the end, useful for padding dialog rounds for test
            self.questions.append("")
            self.answers.append("")

            # image_id serves as key for all three dicts here
            self.captions = {}
            self.dialogs = {}
            self.num_rounds = {}
            for dialog_for_image in visdial_data["data"]["dialogs"]:
                self.captions[dialog_for_image["image_id"]] = dialog_for_image["caption"]

                # record original length of dialog, before padding
                # 10 for train and val splits, 10 or less for test split
                self.num_rounds[dialog_for_image["image_id"]] = len(dialog_for_image["dialog"])

                # pad dialog at the end with empty question and answer pairs (for test split)
                while len(dialog_for_image["dialog"]) < 10:
                    dialog_for_image["dialog"].append({"question": -1, "answer": -1})

                # add empty answer /answer options if not provided (for test split)
                for i in range(len(dialog_for_image["dialog"])):
                    if "answer" not in dialog_for_image["dialog"][i]:
                        dialog_for_image["dialog"][i]["answer"] = -1
                    if "answer_options" not in dialog_for_image["dialog"][i]:
                        dialog_for_image["dialog"][i]["answer_options"] = [-1] * 100

                self.dialogs[dialog_for_image["image_id"]] = dialog_for_image["dialog"]

            print(f"[{self._split}] Tokenizing questions...")
            for i in tqdm(range(len(self.questions))):
                self.questions[i] = word_tokenize(self.questions[i])

            print(f"[{self._split}] Tokenizing answers...")
            for i in tqdm(range(len(self.answers))):
                self.answers[i] = word_tokenize(self.answers[i])

            print(f"[{self._split}] Tokenizing captions...")
            for image_id, caption in tqdm(self.captions.items()):
                self.captions[image_id] = word_tokenize(caption)

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, str, List]]:
        caption_for_image = self.captions[image_id]
        dialog_for_image = copy.copy(self.dialogs[image_id])
        num_rounds = self.num_rounds[image_id]

        # replace question and answer indices with actual word tokens
        for i in range(len(dialog_for_image)):
            dialog_for_image[i]["question"] = self.questions[dialog_for_image[i]["question"]]
            dialog_for_image[i]["answer"] = self.answers[dialog_for_image[i]["answer"]]
            for j, answer_option in enumerate(dialog_for_image[i]["answer_options"]):
                dialog_for_image[i]["answer_options"][j] = self.answers[answer_option]

        return {
            "image_id": image_id,
            "caption": caption_for_image,
            "dialog": dialog_for_image,
            "num_rounds": num_rounds
        }

    @property
    def split(self):
        return self._split


class ImageFeaturesHdfReader(object):
    """
    A reader for generic HDF files with non-nested groups. Here it serves the purpose of reading
    pre-trained image features. A typical HDF file is expected to have a column of primary key
    ("image_id") and one or more columns containing image features.

    Example of an HDF file:
    ```
    visdial_train_faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details about HDF structure.

    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split image features.
    primary_key : str, optional (default = "image_id")
        Name of column in HDF holding the primary key, named "image_id" in all provided
        pre-extracted feature files.
    """

    def __init__(self,
                 features_hdfpath: str,
                 primary_key: str = "image_id"):
        self.features_hdfpath = features_hdfpath
        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            self.features_hdfkeys = list(features_hdf.keys())
            self.primary_key_list = list(features_hdf[primary_key])
            self._split = features_hdf.attrs["split"]

    def __len__(self):
        return len(self.primary_key_list)

    def __getitem__(self, primary_key):
        features_hdf = h5py.File(self.features_hdfpath, "r")
        index = self.primary_key_list.index(primary_key)
        item = {}
        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            for key in self.features_hdfkeys:
                item[key] = features_hdf[key][index]
        return item

    @property
    def split(self):
        return self._split
