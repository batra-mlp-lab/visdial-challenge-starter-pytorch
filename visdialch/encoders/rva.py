import torch
from torch import nn
from torch.nn import functional as F

from visdialch.utils import DynamicRNN
from visdialch.utils import Q_ATT, H_ATT, V_Filter
from .modules import RvA_MODULE


class RvAEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary), 
            config["word_embedding_size"], 
            padding_idx=vocabulary.PAD_INDEX
        )

        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"], 
            bidirectional=True
        )
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"], 
            bidirectional=True
        )        
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)
        
        # self attention for question
        self.Q_ATT_ans = Q_ATT(config)
        self.Q_ATT_ref = Q_ATT(config)
        # question-based history attention
        self.H_ATT_ans = H_ATT(config)

        # modules
        self.RvA_MODULE = RvA_MODULE(config)
        self.V_Filter = V_Filter(config)

        # fusion layer
        self.fusion = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["img_feature_size"] + config["word_embedding_size"] + config["lstm_hidden_size"] * 2, 
                config["lstm_hidden_size"]
            )
        )
        # other useful functions
        self.softmax = nn.Softmax(dim=-1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, batch, return_att=False):
        # img - shape: (batch_size, num_proposals, img_feature_size) - RCNN bottom-up features
        img = batch["img_feat"]
        batch_size = batch["ques"].size(0)
        num_rounds = batch["ques"].size(1)

        # init language embedding
        # ques_word_embed - shape: (batch_size, num_rounds, quen_len_max, word_embedding_size)
        # ques_word_encoded - shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        # ques_not_pad - shape: (batch_size, num_rounds, quen_len_max)
        # ques_encoded - shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_word_embed, ques_word_encoded, ques_not_pad, ques_encoded = self.init_q_embed(batch)
        # hist_word_embed - shape: (batch_size, num_rounds, hist_len_max, word_embedding_size)
        # hist_encoded - shape: (batch_size, num_rounds, lstm_hidden_size)
        hist_word_embed, hist_encoded = self.init_h_embed(batch)
        # cap_word_embed - shape: (batch_size, 1, quen_len_max, word_embedding_size)
        # cap_word_encoded - shape: (batch_size, 1, quen_len_max, lstm_hidden_size)
        # cap_not_pad - shape: (batch_size, 1, quen_len_max)
        cap_word_embed, cap_word_encoded, cap_not_pad = self.init_cap_embed(batch)

        # question feature for RvA
        # ques_ref_feat - shape: (batch_size, num_rounds, word_embedding_size)
        # ques_ref_att - shape: (batch_size, num_rounds, quen_len_max)
        ques_ref_feat, ques_ref_att = self.Q_ATT_ref(ques_word_embed, ques_word_encoded, ques_not_pad)
        # cap_ref_feat - shape: (batch_size, 1, word_embedding_size)
        cap_ref_feat, _ = self.Q_ATT_ref(cap_word_embed, cap_word_encoded, cap_not_pad)

        # RvA module
        ques_feat = (cap_ref_feat, ques_ref_feat, ques_encoded)
        # img_att - shape: (batch_size, num_rounds, num_proposals)
        img_att, att_set = self.RvA_MODULE(img, ques_feat, hist_encoded)
        # img_feat - shape: (batch_size, num_rounds, img_feature_size)
        img_feat = torch.bmm(img_att, img)
        
        # ans_feat for joint embedding
        # hist_ans_feat - shape: (batch_size, num_rounds, lstm_hidden_size*2)
        hist_ans_feat = self.H_ATT_ans(hist_encoded, ques_encoded)
        # ques_ans_feat - shape: (batch_size, num_rounds, lstm_hidden_size)
        # ques_ans_att - shape: (batch_size, num_rounds, quen_len_max)
        ques_ans_feat, ques_ans_att = self.Q_ATT_ans(ques_word_embed, ques_word_encoded, ques_not_pad)
        # img_ans_feat - shape: (batch_size, num_rounds, img_feature_size)
        img_ans_feat = self.V_Filter(img_feat, ques_ans_feat)

        # joint embedding
        fused_vector = torch.cat((img_ans_feat, ques_ans_feat, hist_ans_feat), -1)
        # img_ans_feat - shape: (batch_size, num_rounds, lstm_hidden_size)
        fused_embedding = torch.tanh(self.fusion(fused_vector))

        if return_att:
            return fused_embedding, att_set + (ques_ref_att, ques_ans_att)
        else:
            return fused_embedding

    def init_q_embed(self, batch):
        ques = batch["ques"] # shape: (batch_size, num_rounds, quen_len_max)
        batch_size, num_rounds, _ = ques.size()
        lstm_hidden_size = self.config["lstm_hidden_size"]
        
        # question feature
        ques_not_pad = (ques!=0).float() # shape: (batch_size, num_rounds, quen_len_max)
        ques = ques.view(-1, ques.size(-1)) # shape: (batch_size*num_rounds, quen_len_max)
        ques_word_embed = self.word_embed(ques) # shape: (batch_size*num_rounds, quen_len_max, lstm_hidden_size)
        ques_word_encoded, _ = self.ques_rnn(ques_word_embed, batch['ques_len']) # shape: (batch_size*num_rounds, quen_len_max, lstm_hidden_size*2)
        quen_len_max = ques_word_encoded.size(1)
        loc = batch['ques_len'].view(-1).cpu().numpy()-1
        ques_encoded_forawrd = ques_word_encoded[range(num_rounds*batch_size), loc, :lstm_hidden_size] # shape: (batch_size*num_rounds, lstm_hidden_size) 
        ques_encoded_backward = ques_word_encoded[:, 0, lstm_hidden_size:] # shape: (batch_size*num_rounds, lstm_hidden_size) 
        ques_encoded = torch.cat((ques_encoded_forawrd, ques_encoded_backward), dim=-1)
        ques_encoded = ques_encoded.view(-1, num_rounds, ques_encoded.size(-1)) # shape: (batch_size, num_rounds, lstm_hidden_size*2)
        ques_word_encoded = ques_word_encoded.view(-1, num_rounds, quen_len_max, ques_word_encoded.size(-1)) # shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        ques_word_embed = ques_word_embed.view(-1, num_rounds, quen_len_max, ques_word_embed.size(-1)) # shape: (batch_size, num_rounds, quen_len_max, word_embedding_size)

        return ques_word_embed, ques_word_encoded, ques_not_pad, ques_encoded

    def init_h_embed(self, batch):
        hist = batch["hist"] # shape: (batch_size, num_rounds, hist_len_max)
        batch_size, num_rounds, _ = hist.size()
        lstm_hidden_size = self.config["lstm_hidden_size"]
        
        hist = hist.view(-1, hist.size(-1)) # shape: (batch_size*num_rounds, hist_len_max)
        hist_word_embed = self.word_embed(hist) # shape: (batch_size*num_rounds, hist_len_max, word_embedding_size)
        hist_word_encoded, _ = self.hist_rnn(hist_word_embed, batch['hist_len']) # shape: (batch_size*num_rounds, hist_len_max, lstm_hidden_size*2)
        loc = batch['hist_len'].view(-1).cpu().numpy()-1
        hist_encoded_forward = hist_word_encoded[range(num_rounds*batch_size), loc, :lstm_hidden_size] # shape: (batch_size*num_rounds, hist_len_max, lstm_hidden_size*2) 
        hist_encoded_backward = hist_word_encoded[:, 0, lstm_hidden_size:] # shape: (batch_size*num_rounds, lstm_hidden_size) 
        hist_encoded = torch.cat((hist_encoded_forward, hist_encoded_backward), dim=-1)
        hist_encoded = hist_encoded.view(-1, num_rounds, hist_encoded.size(-1)) # shape: (batch_size, num_rounds, lstm_hidden_size)

        return hist_word_embed, hist_encoded

    def init_cap_embed(self, batch):
        cap = batch["hist"][:, :1, :] # shape: (batch_size, 1, hist_len_max)

        # caption feature like question
        cap_not_pad = (cap!=0).float() # shape: (batch_size, 1, hist_len_max)
        cap_word_embed = self.word_embed(cap.squeeze(1)) # shape: (batch_size*1, hist_len_max, word_embedding_size)
        cap_len = batch['hist_len'][:, :1]
        cap_word_encoded, _ = self.ques_rnn(cap_word_embed, cap_len) # shape: (batch_size*1, hist_len_max, lstm_hidden_size)
        cap_word_encoded = cap_word_encoded.unsqueeze(1) # shape: (batch_size, 1, hist_len_max, lstm_hidden_size)
        cap_word_embed = cap_word_embed.unsqueeze(1) # shape: (batch_size, 1, hist_len_max, word_embedding_size)

        return cap_word_embed, cap_word_encoded, cap_not_pad
