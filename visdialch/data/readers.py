"""
A Reader simply reads data from disk and returns it almost as is. Readers should be utilized by 
torch ``Dataset``s. Any type of data pre-processing is not recommended in the reader, such as
tokenizing sentences, embedding tokens, or passing an image through a pre-trained CNN.

Each Reader should be initialized by one or more file paths, and should provide access to a
single data instance by ``image_id`` of VisDial images (implement ``__getitem__``). In addition,
a reader should implement ``close`` method, which closes any open file handles, and explicitly
frees memory by deleting variables. A Reader shall become useless after calling ``close``.
"""

import copy
import json
from typing import Dict, List, Union


class VisDialJsonReader(object):
    """
    A simple reader for VisDial v1.0 data. The json file must have the same structure as mentioned
    on ``https://visualdialog.org/data``.

    Parameters
    ----------
    visdial_json : str
        Path to a json file containing VisDial v1.0 train, val or test data.
    """
    def __init__(self, visdial_json: str):
        with open(visdial_json, "r") as visdial_file:
            visdial_data = json.load(visdial_file)
            self.questions = visdial_data["data"]["questions"]
            self.answers = visdial_data["data"]["answers"]

            self.captions = {}
            self.dialogs = {}
            for dialog_for_image in visdial_data["data"]["dialogs"]:
                self.captions[dialog_for_image["image_id"]] = dialog_for_image["caption"]
                self.dialogs[dialog_for_image["image_id"]] = dialog_for_image["dialog"]
            self._split = visdial_data["split"]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, str, List]]:
        caption_for_image = self.captions[image_id]
        dialog_for_image = copy.copy(self.dialogs[image_id])

        # replace question and answer indices with actual string
        for i in range(len(dialog_for_image)):
            dialog_for_image[i]["question"] = self.questions[dialog_for_image[i]["question"]]

            # last round in test split does not have answer, add empty string
            if "answer" in dialog_for_image[i]:
                dialog_for_image[i]["answer"] = self.answers[dialog_for_image[i]["answer"]]
            else:
                dialog_for_image[i]["answer"] = ""

            # only the last round has answer options (for test split)
            if "answer_options" in dialog_for_image[i]:
                for j, answer_option in enumerate(dialog_for_image[i]["answer_options"]):
                    dialog_for_image[i]["answer_options"][j] = self.answers[answer_option]

        # pad dialog at the end with empty question and answer pairs (for test split)
        while len(dialog_for_image) < 10:
            dialog_for_image.append({"question": "", "answer": ""})

        return {
            "image_id": image_id,
            "caption": caption_for_image,
            "dialog": dialog_for_image
        }

    @property
    def split(self):
        return self._split

    def close(self):
        """Delete all the data this reader is holding, and free memory. The reader becomes
        useless after calling this method.
        """
        del self.questions, self.answers, self.captions, self.dialogs
