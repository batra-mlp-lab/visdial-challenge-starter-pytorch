import argparse
from tqdm import tqdm
from typing import Any, Dict
import json


class SubsetExtractor:
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.num_dialogs = config.num_dialogs

    def extract_subset(self, type: str):
        data_path = self._get_json_path(type)
        data_json = json.load(open(data_path))

        # print(data_json.keys()) # ['version', 'split', 'data']
        # print(data_json['data'].keys()) # ['questions', 'dialogs', 'answers']

        # SA: todo subset only relevant questions and answers for speed up
        questions = data_json['data']['questions']  # All questions
        answers = data_json['data']['answers']  # All answers
        dialogs = data_json['data']['dialogs'][0:self.num_dialogs]

        dialog_ques = set()  # All questions
        dialog_ans = set()  # Answer options and correct answers
        for dialog in dialogs:
            # print(dialog)
            dialog_obj = dialog["dialog"]
            for turn in dialog_obj:
                # dialog_ques.add(dialog['dialog']['question'])
                # dialog_ans.add(dialog['dialog']['answer'])
                # dialog_ans.update(dialog['dialog']['answer_options'])
                dialog_ques.add(turn['question'])
                dialog_ans.add(turn['answer'])
                dialog_ans.update(turn['answer_options'])

        print("Subsetting num of questions to {} from {}".format(max(dialog_ques),len(questions)))
        print("Subsetting num of answers to {} from {}".format(max(dialog_ans),len(answers)))

        questions = questions[:max(dialog_ques)]
        answers = answers[:max(dialog_ans)]

        data = {
            "questions": questions,
            "answers": answers,
            "dialogs": dialogs
        }

        out_json = {
            "version": data_json['version'],
            "split": data_json['split'],
            "data": data
        }

        save_file_path = self._save_file_path(type)
        with open(save_file_path, 'w') as outfile:
            json.dump(out_json, outfile)

    def _get_json_path(self, type: str,
                       split: str = '1.0') -> str:
        json_path = "{}/visdial_{}_{}.json".format(self.data_dir, split, type)
        return json_path

    def _save_file_path(self, type: str,
                        split: str = '1.0') -> str:
        file_path = "{}/visdial_{}_{}_debug.json".format(self.data_dir, split, type)
        return file_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", default="data/",
        help="Path to data directory with json dialogues."
    )
    parser.add_argument(
        "-n", "--num_dialogs", default=5,
        help="Number of dialogs to keep a subset."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    extractor = SubsetExtractor(config)
    extractor.extract_subset("train")
    extractor.extract_subset("val")
    extractor.extract_subset("test")
