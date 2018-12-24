from typing import Dict, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from visdialch.data.readers import VisDialJsonReader
from visdialch.data.vocabulary import Vocabulary


class VisDialDataset(Dataset):
    def __init__(self,
                 visdial_json_filepath: str,
                 config: Dict[str, Union[int, str]],
                 overfit: bool = False):
        super().__init__()
        self.config = config
        self.reader = VisDialJsonReader(visdial_json_filepath)
        self.vocabulary = Vocabulary(
            config["word_counts_json"], min_count=config["vocab_min_count"]
        )

        # keep a list of image_ids as primary keys to access data
        self.image_ids = list(self.reader.dialogs.keys())

        # print("Dataloader loading h5 file: {}".format(config["img_features_h5"]))
        # img_file = h5py.File(config["img_features_h5"], "r")
        # img_feats = torch.from_numpy(np.array(img_file["images_" + dtype]))

        # if config["img_norm"]:
        #     print("Normalizing image features...")
        #     img_feats = F.normalize(img_feats, dim=1, p=2)

        # reduce amount of data for preprocessing in fast mode
        if overfit:
            self.image_ids = self.image_ids[:5]

    @property
    def split(self):
        return self.reader.split

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # get image_id, which serves as a primary key for current instance
        image_id = self.image_ids[index]

        # retrieve instance for this image_id using the reader
        visdial_instance = self.reader[image_id]
        caption = visdial_instance["caption"]
        dialog = visdial_instance["dialog"]

        # tokenize caption, question, answer and answer options to integers, using Vocabulary
        caption = [self.vocabulary.get_index_by_word(word) for word in caption]
        for i in range(len(dialog)):
            dialog[i]["question"] = [
                self.vocabulary.get_index_by_word(word) for word in dialog[i]["question"]
            ]
            dialog[i]["answer"] = [
                self.vocabulary.get_index_by_word(word) for word in dialog[i]["answer"]
            ]
            for j in range(len(dialog[i]["answer_options"])):
                dialog[i]["answer_options"][j] = [
                    self.vocabulary.get_index_by_word(word)
                    for word in dialog[i]["answer_options"][j]
                ]

        questions, question_lengths = self._pad_sequences(
            [dialog_round["question"] for dialog_round in dialog]
        )
        history, history_lengths = self._get_history(
            caption,
            [dialog_round["question"] for dialog_round in dialog],
            [dialog_round["answer"] for dialog_round in dialog]
        )

        answer_options = []
        answer_option_lengths = []
        for dialog_round in dialog:
            options, option_lengths = self._pad_sequences(dialog_round["answer_options"])
            answer_options.append(options)
            answer_option_lengths.append(option_lengths)
        answer_options = torch.stack(answer_options, 0)

        if "test" not in self.split:
            answer_indices = [dialog_round["gt_index"] for dialog_round in dialog]

        # collect everything as tensors for ``collate_fn`` of dataloader to work seemlessly
        # questions, history, etc. are converted to LongTensors, for nn.Embedding input
        item = {}
        item["img_ids"] = torch.tensor(image_id).long()
        item["ques"] = questions.long()
        item["hist"] = history.long()
        item["opt"] = answer_options.long()
        item["ques_len"] = torch.tensor(question_lengths).long()
        item["hist_len"] = torch.tensor(history_lengths).long()
        item["opt_len"] = torch.tensor(answer_option_lengths).long()
        item["num_rounds"] = torch.tensor(visdial_instance["num_rounds"]).long()
        if "test" not in self.split:
            item["ans_inds"] = torch.tensor(answer_indices).long()
        # TODO: add image features in here, for now put a random tensor
        item["img_feat"] = torch.randn(2048)
        return item

    def _pad_sequences(self, sequences: List[List[int]]):
        """Given tokenized sequences (either questions, answers or answer options, tokenized
        in ``__getitem__``), padding them to maximum specified sequence length. Return as a
        tensor of size ``(*, max_sequence_length)``.

        This method is only called in ``__getitem__``, chunked out separately for readability.

        Parameters
        ----------
        sequences : List[List[int]]
            List of tokenized sequences, each sequence is typically a List[int].

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensor of sequences padded to max length, and length of sequences before padding.
        """

        for i in range(len(sequences)):
            sequences[i] = sequences[i][: self.config["max_sequence_length"] - 1]
            # <S> is not necessary this will be simply encoded, without generation
            sequences[i].append(self.vocabulary.EOS_INDEX)
        sequence_lengths = [len(sequence) for sequence in sequences]

        # pad all sequences to max_sequence_length
        maxpadded_sequences = torch.full(
            (len(sequences), self.config["max_sequence_length"]),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_sequences = pad_sequence(
            [torch.tensor(sequence) for sequence in sequences],
            batch_first=True, padding_value=self.vocabulary.PAD_INDEX
        )
        maxpadded_sequences[:, :padded_sequences.size(1)] = padded_sequences
        return maxpadded_sequences, sequence_lengths

    def _get_history(self,
                     caption: List[int],
                     questions: List[List[int]],
                     answers: List[List[int]]):
        # allow double length of caption, equivalent to a concatenated QA pair
        caption = caption[: self.config["max_sequence_length"] * 2 - 1]
        caption.append(self.vocabulary.EOS_INDEX)

        for i in range(len(questions)):
            questions[i] = questions[i][: self.config["max_sequence_length"] - 1]
            questions[i].append(self.vocabulary.EOS_INDEX)

        for i in range(len(answers)):
            answers[i] = answers[i][: self.config["max_sequence_length"] - 1]
            answers[i].append(self.vocabulary.EOS_INDEX)

        # history for first round is caption, else concatenated QA pair of previous round
        history = []
        history.append(caption)
        for question, answer in zip(questions, answers):
            history.append(question + answer)
        # drop last entry from history (there's no eleventh question)
        history = history[:-1]
        max_history_length = self.config["max_sequence_length"] * 2

        if self.config["concat_history"]:
            # concatenated_history has similar structure as history, except it contains
            # concatenated QA pairs from previous rounds
            concatenated_history = []
            concatenated_history.append(caption)
            for i in range(1, len(history)):
                concatenated_history.append([])
                for j in range(i + 1):
                    concatenated_history[i].extend(history[j])

            max_history_length = self.config["max_sequence_length"] * 2 * len(history)
            history = concatenated_history

        history_lengths = [len(round_history) for round_history in history]
        maxpadded_history = torch.full(
            (len(history), max_history_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_history = pad_sequence(
            [torch.tensor(round_history) for round_history in history],
            batch_first=True, padding_value=self.vocabulary.PAD_INDEX
        )
        maxpadded_history[:, :padded_history.size(1)] = padded_history
        return maxpadded_history, history_lengths
