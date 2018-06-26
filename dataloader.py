import os
import json
from six import iteritems

import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class VisDialDataset(Dataset):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Dataloader specific arguments')
        parser.add_argument('-input_img', default='data/data_img.h5',
                                help='HDF5 file with image features')
        parser.add_argument('-input_ques', default='data/visdial_data.h5',
                                help='HDF5 file with preprocessed questions')
        parser.add_argument('-input_json', default='data/visdial_params.json',
                                help='JSON file with image paths and vocab')
        parser.add_argument('-img_norm', default=1, choices=[1, 0],
                                help='normalize the image feature. 1=yes, 0=no')
        return parser

    def __init__(self, opt, subsets):
        """Initialize the dataset with splits given by 'subsets', where
        subsets is taken from ['train', 'val', 'test']
        """
        super().__init__()
        self.opt = opt
        self.subsets = tuple(subsets)

        print("\nDataloader loading json file: {}".format(opt.input_json))
        with open(opt.input_json, 'r') as info_file:
            info = json.load(info_file)
            # possible keys: {'ind2word', 'word2ind', 'unique_img_(split)'}
            for key, value in iteritems(info):
                setattr(self, key, value)

        # add <START> and <END> to vocabulary
        word_count = len(self.word2ind)
        self.word2ind['<START>'] = word_count + 1
        self.word2ind['<END>'] = word_count + 2
        self.start_token = self.word2ind['<START>']
        self.end_token = self.word2ind['<END>']

        # padding + <START> + <END> token
        self.vocab_size = word_count + 3
        print("Vocab size with <START>, <END>: {}".format(self.vocab_size))

        # construct reverse of word2ind after adding tokens
        self.ind2word = {
            int(ind): word
            for word, ind in iteritems(self.word2ind)
        }

        print("Dataloader loading h5 file: {}".format(opt.input_ques))
        ques_file = h5py.File(opt.input_ques, 'r')

        print("Dataloader loading h5 file: {}".format(opt.input_img))
        img_file = h5py.File(opt.input_img, 'r')

        # load all data mats from ques_file into this
        self.data = {}

        # map from load to save labels
        io_map = {
            'ques_{}': '{}_ques',
            'ques_length_{}': '{}_ques_len',
            'ans_{}': '{}_ans',
            'ans_length_{}': '{}_ans_len',
            'img_pos_{}': '{}_img_pos',
            'opt_{}': '{}_opt',
            'opt_length_{}': '{}_opt_len',
            'opt_list_{}': '{}_opt_list',
            'num_rounds_{}': '{}_num_rounds',
            'ans_index_{}': '{}_ans_ind'
        }

        # processing every split in subsets
        for dtype in subsets:  # dtype is in ['train', 'val', 'test']
            print("\nProcessing split [{}]...".format(dtype))
            # read the question, answer, option related information
            for load_label, save_label in iteritems(io_map):
                if load_label.format(dtype) not in ques_file:
                    continue
                self.data[save_label.format(dtype)] = torch.from_numpy(
                    np.array(ques_file[load_label.format(dtype)], dtype='int64'))

            print("Reading image features...")
            img_feats = torch.from_numpy(np.array(img_file['images_' + dtype]))

            if opt.img_norm:
                print("Normalizing image features...")
                img_feats = F.normalize(img_feats, dim=1, p=2)

            # save image features
            self.data['{}_img_fv'.format(dtype)] = img_feats
            img_fnames = getattr(self, 'unique_img_{}'.format(dtype))
            self.data['{}_img_fnames'.format(dtype)] = img_fnames

            # read the history
            caption_map = {
                'cap_{}': '{}_cap',
                'cap_length_{}': '{}_cap_len'
            }
            for load_label, save_label in iteritems(caption_map):
                mat = np.array(ques_file[load_label.format(dtype)], dtype='int32')
                self.data[save_label.format(dtype)] = torch.from_numpy(mat)

        self.num_data_points = {}
        for dtype in subsets:
            self.num_data_points[dtype] = len(getattr(self, 'unique_img_{}'.format(dtype)))

        # prepare dataset for training
        for dtype in subsets:
            self._process_questions(dtype)
            self._process_history(dtype)
            self._process_options(dtype)
            self._process_answers(dtype)

            # 1 indexed to 0 indexed
            self.data[dtype + '_opt'] -= 1
            if dtype + '_ans_ind' in self.data:
                self.data[dtype + '_ans_ind'] -= 1

        # default pytorch loader dtype is set to train
        if 'train' in subsets:
            self._split = 'train'
        else:
            self._split = subsets[0]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets  # ['train', 'val', 'test']
        self._split = split

    def _process_questions(self, dtype):
        """Right align questions."""
        print("Right aligning questions for [{}]...".format(dtype))
        self.data[dtype + '_ques_fwd'] = self._right_align(
            self.data[dtype + '_ques'], self.data[dtype + '_ques_len'])

    def _process_history(self, dtype):
        """Process caption as well as right align history.
        Optionally, concatenate history for lf-encoder.
        """
        captions = self.data[dtype + '_cap']
        questions = self.data[dtype + '_ques']
        ques_len = self.data[dtype + '_ques_len']
        cap_len = self.data[dtype + '_cap_len']
        max_ques_len = questions.size(2)

        answers = self.data[dtype + '_ans']
        ans_len = self.data[dtype + '_ans_len']
        num_convs, num_rounds, max_ans_len = answers.size()

        if self.opt.concat_history:
            self.max_hist_len = min(num_rounds * (max_ques_len + max_ans_len), 300)
            history = torch.zeros(num_convs, num_rounds, self.max_hist_len).long()
        else:
            history = torch.zeros(num_convs, num_rounds, max_ques_len + max_ans_len).long()
        hist_len = torch.zeros(num_convs, num_rounds).long()

        # go over each question and append it with answer
        for th_id in range(num_convs):
            clen = cap_len[th_id]
            hlen = min(clen, max_ques_len + max_ans_len)
            for round_id in range(num_rounds):
                if round_id == 0:
                    # first round has caption as history
                    history[th_id][round_id][:max_ques_len + max_ans_len] \
                        = captions[th_id][:max_ques_len + max_ans_len]
                else:
                    qlen = ques_len[th_id][round_id - 1]
                    alen = ans_len[th_id][round_id - 1]
                    # if concat_history, string together all previous question-answer pairs
                    if self.opt.concat_history:
                        history[th_id][round_id][:hlen] = history[th_id][round_id - 1][:hlen]
                        history[th_id][round_id][hlen] = self.word2ind['<END>']
                        if qlen > 0:
                            history[th_id][round_id][hlen:hlen + qlen] \
                                = questions[th_id][round_id - 1][:qlen]
                        if alen > 0:
                            # print(round_id, history[th_id][round_id][:10], answers[th_id][round_id][:10])
                            history[th_id][round_id][hlen + qlen:hlen + qlen + alen] \
                                = answers[th_id][round_id - 1][:alen]
                        hlen = hlen + qlen + alen
                    # else, history is just previous round question-answer pair
                    else:
                        if qlen > 0:
                            history[th_id][round_id][:qlen] = questions[th_id][round_id - 1][:qlen]
                        if alen > 0:
                            history[th_id][round_id][qlen:qlen + alen] \
                                = answers[th_id][round_id - 1][:alen]
                        hlen = alen + qlen
                # save the history length
                hist_len[th_id][round_id] = hlen

        # right align history and then save
        print("Right aligning history for [{}]...".format(dtype))
        self.data[dtype + '_hist'] = self._right_align(history, hist_len)
        self.data[dtype + '_hist_len'] = hist_len

    def _process_answers(self, dtype):
        """Prefix answers with <START>, <END>, adjust answer lengths."""
        answers = self.data[dtype + '_ans']
        ans_len = self.data[dtype + '_ans_len']
        num_convs, num_rounds, max_ans_len = answers.size()

        decode_in = torch.zeros(num_convs, num_rounds, max_ans_len + 1).long()
        decode_out = torch.zeros(num_convs, num_rounds, max_ans_len + 1).long()
        
        # decode_in begins with <START>
        decode_in[:, :, 0] = self.word2ind['<START>']

        # go over each answer and modify
        end_token_id = self.word2ind['<END>']
        for th_id in range(num_convs):
            for round_id in range(num_rounds):
                length = ans_len[th_id][round_id]
                if length > 0:
                    decode_in[th_id][round_id][1:length + 1] = answers[th_id][round_id][:length]
                    decode_out[th_id][round_id][:length] = answers[th_id][round_id][:length]
                else:
                    if dtype != 'test':
                        print("Warning: empty answer at ({0} {1} {2})".format(
                                th_id, round_id, length))
            decode_out[th_id][round_id][length] = end_token_id

        self.data[dtype + '_ans_len'] += 1
        self.data[dtype + '_ans_in'] = decode_in
        self.data[dtype + '_ans_out'] = decode_out

    def _process_options(self, dtype):
        lengths = self.data[dtype + '_opt_len']
        answers = self.data[dtype + '_ans']
        ans_len = self.data[dtype + '_ans_len']
        num_convs, _, max_ans_len = answers.size()

        options = self.data[dtype + '_opt_list']
        opt_list_len = options.size(0)
        decode_in = torch.zeros(opt_list_len, max_ans_len + 1).long()
        decode_out = torch.zeros(opt_list_len, max_ans_len + 1).long()

        # decodeIn begins with <START>
        decode_in[:, 0] = self.word2ind['<START>']

        # go over each answer and modify
        end_token_id = self.word2ind['<END>']
        for opt_id in tqdm(range(opt_list_len)):
            length = lengths[opt_id]
            if length > 0:
                decode_in[opt_id][:length + 1] = options[opt_id][:length]
                decode_out[opt_id][:length] = options[opt_id][:length]
                decode_out[opt_id][length] = end_token_id
            else:
                if dtype != 'test':
                    print("Warning: empty answer for {0} at {1}".format(dtype, opt_id))

        self.data[dtype + '_opt_len'] = self.data[dtype + '_opt_len'] + 1
        self.data[dtype + '_opt_in'] = decode_in
        self.data[dtype + '_opt_out'] = decode_out

    @staticmethod
    def _right_align(sequences, lengths):
        """Right align the question/history tokens in 3d volume."""
        raligned = sequences.clone().fill_(0)
        num_dims = sequences.dim()

        if num_dims == 3:
            num_imgs, num_ques, max_ques_len = sequences.size()
            for img_id in tqdm(range(num_imgs)):  # total images
                for ques_id in range(num_ques):  # total questions per image
                    # do only for non zero sequence counts
                    if lengths[img_id][ques_id] > 0:
                        raligned[img_id][ques_id][
                            (max_ques_len - lengths[img_id][ques_id]):] = \
                            sequences[img_id][ques_id][:lengths[img_id][ques_id]]
        elif num_dims == 2:
            # handle 2-dimensional matrices as well
            num_imgs, max_ques_len = sequences.size()
            for img_id in tqdm(range(num_imgs)):
                # do only for non zero sequence counts
                if lengths[img_id] > 0:
                    raligned[img_id][(max_ques_len - lengths[img_id]):] = \
                        sequences[img_id][:lengths[img_id]]
        return raligned
