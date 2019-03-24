from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from visdialch.data.readers import (
    DialogsReader,
    DenseAnnotationsReader,
    ImageFeaturesHdfReader,
)
from visdialch.data.vocabulary import Vocabulary


class VisDialDataset(Dataset):
    """
    A full representation of VisDial v1.0 (train/val/test) dataset. According
    to the appropriate split, it returns dictionary of question, image,
    history, ground truth answer, answer options, dense annotations etc.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dialogs_jsonpath: str,
        dense_annotations_jsonpath: Optional[str] = None,
        overfit: bool = False,
        in_memory: bool = False,
        return_options: bool = True,
        add_boundary_toks: bool = False,
    ):
        super().__init__()
        self.config = config
        self.return_options = return_options
        self.add_boundary_toks = add_boundary_toks
        self.dialogs_reader = DialogsReader(dialogs_jsonpath)

        if "val" in self.split and dense_annotations_jsonpath is not None:
            self.annotations_reader = DenseAnnotationsReader(
                dense_annotations_jsonpath
            )
        else:
            self.annotations_reader = None

        self.vocabulary = Vocabulary(
            config["word_counts_json"], min_count=config["vocab_min_count"]
        )

        # Initialize image features reader according to split.
        image_features_hdfpath = config["image_features_train_h5"]
        if "val" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_val_h5"]
        elif "test" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_test_h5"]

        self.hdf_reader = ImageFeaturesHdfReader(
            image_features_hdfpath, in_memory
        )

        # Keep a list of image_ids as primary keys to access data.
        self.image_ids = list(self.dialogs_reader.dialogs.keys())
        if overfit:
            self.image_ids = self.image_ids[:5]

    @property
    def split(self):
        return self.dialogs_reader.split

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # Get image_id, which serves as a primary key for current instance.
        image_id = self.image_ids[index]

        # Get image features for this image_id using hdf reader.
        image_features = self.hdf_reader[image_id]
        image_features = torch.tensor(image_features)
        # Normalize image features at zero-th dimension (since there's no batch
        # dimension).
        if self.config["img_norm"]:
            image_features = normalize(image_features, dim=0, p=2)

        # Retrieve instance for this image_id using json reader.
        visdial_instance = self.dialogs_reader[image_id]
        caption = visdial_instance["caption"]
        dialog = visdial_instance["dialog"]

        # Convert word tokens of caption, question, answer and answer options
        # to integers.
        caption = self.vocabulary.to_indices(caption)
        for i in range(len(dialog)):
            dialog[i]["question"] = self.vocabulary.to_indices(
                dialog[i]["question"]
            )
            if self.add_boundary_toks:
                dialog[i]["answer"] = self.vocabulary.to_indices(
                    [self.vocabulary.SOS_TOKEN]
                    + dialog[i]["answer"]
                    + [self.vocabulary.EOS_TOKEN]
                )
            else:
                dialog[i]["answer"] = self.vocabulary.to_indices(
                    dialog[i]["answer"]
                )

            if self.return_options:
                for j in range(len(dialog[i]["answer_options"])):
                    if self.add_boundary_toks:
                        dialog[i]["answer_options"][
                            j
                        ] = self.vocabulary.to_indices(
                            [self.vocabulary.SOS_TOKEN]
                            + dialog[i]["answer_options"][j]
                            + [self.vocabulary.EOS_TOKEN]
                        )
                    else:
                        dialog[i]["answer_options"][
                            j
                        ] = self.vocabulary.to_indices(
                            dialog[i]["answer_options"][j]
                        )

        questions, question_lengths = self._pad_sequences(
            [dialog_round["question"] for dialog_round in dialog]
        )
        history, history_lengths = self._get_history(
            caption,
            [dialog_round["question"] for dialog_round in dialog],
            [dialog_round["answer"] for dialog_round in dialog],
        )
        answers_in, answer_lengths = self._pad_sequences(
            [dialog_round["answer"][:-1] for dialog_round in dialog]
        )
        answers_out, _ = self._pad_sequences(
            [dialog_round["answer"][1:] for dialog_round in dialog]
        )

        # Collect everything as tensors for ``collate_fn`` of dataloader to
        # work seamlessly questions, history, etc. are converted to
        # LongTensors, for nn.Embedding input.
        item = {}
        item["img_ids"] = torch.tensor(image_id).long()
        item["img_feat"] = image_features
        item["ques"] = questions.long()
        item["hist"] = history.long()
        item["ans_in"] = answers_in.long()
        item["ans_out"] = answers_out.long()
        item["ques_len"] = torch.tensor(question_lengths).long()
        item["hist_len"] = torch.tensor(history_lengths).long()
        item["ans_len"] = torch.tensor(answer_lengths).long()
        item["num_rounds"] = torch.tensor(
            visdial_instance["num_rounds"]
        ).long()

        if self.return_options:
            if self.add_boundary_toks:
                answer_options_in, answer_options_out = [], []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        [
                            option[:-1]
                            for option in dialog_round["answer_options"]
                        ]
                    )
                    answer_options_in.append(options)

                    options, _ = self._pad_sequences(
                        [
                            option[1:]
                            for option in dialog_round["answer_options"]
                        ]
                    )
                    answer_options_out.append(options)

                    answer_option_lengths.append(option_lengths)
                answer_options_in = torch.stack(answer_options_in, 0)
                answer_options_out = torch.stack(answer_options_out, 0)

                item["opt_in"] = answer_options_in.long()
                item["opt_out"] = answer_options_out.long()
                item["opt_len"] = torch.tensor(answer_option_lengths).long()
            else:
                answer_options = []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        dialog_round["answer_options"]
                    )
                    answer_options.append(options)
                    answer_option_lengths.append(option_lengths)
                answer_options = torch.stack(answer_options, 0)

                item["opt"] = answer_options.long()
                item["opt_len"] = torch.tensor(answer_option_lengths).long()

            if "test" not in self.split:
                answer_indices = [
                    dialog_round["gt_index"] for dialog_round in dialog
                ]
                item["ans_ind"] = torch.tensor(answer_indices).long()

        # Gather dense annotations.
        if "val" in self.split:
            dense_annotations = self.annotations_reader[image_id]
            item["gt_relevance"] = torch.tensor(
                dense_annotations["gt_relevance"]
            ).float()
            item["round_id"] = torch.tensor(
                dense_annotations["round_id"]
            ).long()

        return item

    def _pad_sequences(self, sequences: List[List[int]]):
        """Given tokenized sequences (either questions, answers or answer
        options, tokenized in ``__getitem__``), padding them to maximum
        specified sequence length. Return as a tensor of size
        ``(*, max_sequence_length)``.

        This method is only called in ``__getitem__``, chunked out separately
        for readability.

        Parameters
        ----------
        sequences : List[List[int]]
            List of tokenized sequences, each sequence is typically a
            List[int].

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensor of sequences padded to max length, and length of sequences
            before padding.
        """

        for i in range(len(sequences)):
            sequences[i] = sequences[i][
                : self.config["max_sequence_length"] - 1
            ]
        sequence_lengths = [len(sequence) for sequence in sequences]

        # Pad all sequences to max_sequence_length.
        maxpadded_sequences = torch.full(
            (len(sequences), self.config["max_sequence_length"]),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_sequences = pad_sequence(
            [torch.tensor(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX,
        )
        maxpadded_sequences[:, : padded_sequences.size(1)] = padded_sequences
        return maxpadded_sequences, sequence_lengths

    def _get_history(
        self,
        caption: List[int],
        questions: List[List[int]],
        answers: List[List[int]],
    ):
        # Allow double length of caption, equivalent to a concatenated QA pair.
        caption = caption[: self.config["max_sequence_length"] * 2 - 1]

        for i in range(len(questions)):
            questions[i] = questions[i][
                : self.config["max_sequence_length"] - 1
            ]

        for i in range(len(answers)):
            answers[i] = answers[i][: self.config["max_sequence_length"] - 1]

        # History for first round is caption, else concatenated QA pair of
        # previous round.
        history = []
        history.append(caption)
        for question, answer in zip(questions, answers):
            history.append(question + answer + [self.vocabulary.EOS_INDEX])
        # Drop last entry from history (there's no eleventh question).
        history = history[:-1]
        max_history_length = self.config["max_sequence_length"] * 2

        if self.config.get("concat_history", False):
            # Concatenated_history has similar structure as history, except it
            # contains concatenated QA pairs from previous rounds.
            concatenated_history = []
            concatenated_history.append(caption)
            for i in range(1, len(history)):
                concatenated_history.append([])
                for j in range(i + 1):
                    concatenated_history[i].extend(history[j])

            max_history_length = (
                self.config["max_sequence_length"] * 2 * len(history)
            )
            history = concatenated_history

        history_lengths = [len(round_history) for round_history in history]
        maxpadded_history = torch.full(
            (len(history), max_history_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_history = pad_sequence(
            [torch.tensor(round_history) for round_history in history],
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX,
        )
        maxpadded_history[:, : padded_history.size(1)] = padded_history
        return maxpadded_history, history_lengths
