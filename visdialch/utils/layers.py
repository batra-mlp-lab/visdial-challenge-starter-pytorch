import torch
from torch import nn
from torch.nn import functional as F


class GatedTrans(nn.Module):
    """docstring for GatedTrans"""
    def __init__(self, in_dim, out_dim):
        super(GatedTrans, self).__init__()
        
        self.embed_y = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Tanh()
        )
        self.embed_g = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        x_y = self.embed_y(x_in)
        x_g = self.embed_g(x_in)
        x_out = x_y*x_g

        return x_out


class Q_ATT(nn.Module):
    """Self attention module of questions."""
    def __init__(self, config):
        super(Q_ATT, self).__init__()

        self.embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_hidden_size"]*2,
                config["lstm_hidden_size"]
            ),
        )        
        self.att = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_hidden_size"],
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, ques_word, ques_word_encoded, ques_not_pad):
        # ques_word shape: (batch_size, num_rounds, quen_len_max, word_embed_dim)
        # ques_embed shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size * 2)
        # ques_not_pad shape: (batch_size, num_rounds, quen_len_max)
        # output: img_att (batch_size, num_rounds, embed_dim)
        batch_size = ques_word.size(0)
        num_rounds = ques_word.size(1)
        quen_len_max = ques_word.size(2)

        ques_embed = self.embed(ques_word_encoded)  # shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        ques_norm = F.normalize(ques_embed, p=2, dim=-1)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim)
        
        att = self.att(ques_norm).squeeze(-1) # shape: (batch_size, num_rounds, quen_len_max)
        # ignore <pad> word
        att = self.softmax(att)
        att = att*ques_not_pad # shape: (batch_size, num_rounds, quen_len_max)
        att = att / torch.sum(att, dim=-1, keepdim=True) # shape: (batch_size, num_rounds, quen_len_max)
        feat = torch.sum(att.unsqueeze(-1) * ques_word, dim=-2) # shape: (batch_size, num_rounds, word_embed_dim)
        
        return feat, att

class H_ATT(nn.Module):
    """question-based history attention"""
    def __init__(self, config):
        super(H_ATT, self).__init__()

        self.H_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_hidden_size"]*2,
                config["lstm_hidden_size"]
            ),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_hidden_size"]*2,
                config["lstm_hidden_size"]
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_hidden_size"],
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, hist, ques):
        # hist shape: (batch_size, num_rounds, rnn_dim)
        # ques shape: (batch_size, num_rounds, rnn_dim)
        # output: hist_att (batch_size, num_rounds, embed_dim)
        batch_size = ques.size(0)
        num_rounds = ques.size(1)
        
        hist_embed = self.H_embed(hist) # shape: (batch_size, num_rounds, embed_dim)
        hist_embed = hist_embed.unsqueeze(1).repeat(1, num_rounds, 1, 1) # shape: (batch_size, num_rounds, num_rounds, embed_dim)
        
        ques_embed = self.Q_embed(ques) # shape: (batch_size, num_rounds, embed_dim)
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_rounds, 1) # shape: (batch_size, num_rounds, num_rounds, embed_dim)
        
        att_embed = F.normalize(hist_embed*ques_embed, p=2, dim=-1) # (batch_size, num_rounds, num_rounds, embed_dim)
        att_embed = self.att(att_embed).squeeze(-1)
        att = self.softmax(att_embed) # shape: (batch_size, num_rounds, num_rounds)
        att_not_pad = torch.tril(torch.ones(size=[num_rounds, num_rounds], requires_grad=False)) # shape: (num_rounds, num_rounds)
        att_not_pad = att_not_pad.cuda()
        att_masked = att*att_not_pad # shape: (batch_size, num_rounds, num_rounds) 
        att_masked = att_masked / torch.sum(att_masked, dim=-1, keepdim=True) # shape: (batch_size, num_rounds, num_rounds)
        feat = torch.sum(att_masked.unsqueeze(-1) * hist.unsqueeze(1), dim=-2) # shape: (batch_size, num_rounds, rnn_dim)
        
        return feat

class V_Filter(nn.Module):
    """docstring for V_Filter"""
    def __init__(self, config):
        super(V_Filter, self).__init__()

        self.filter = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["word_embedding_size"], 
                config["img_feature_size"]
            ),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        
    def forward(self, img, ques):
        # img shape: (batch_size, num_rounds, i_dim)
        # ques shape: (batch_size, num_rounds, q_dim)
        # output: img_att (batch_size, num_rounds, embed_dim)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        ques_embed = self.filter(ques) # shape: (batch_size, num_rounds, embed_dim)
        
        # gated
        img_fused = img * ques_embed

        return img_fused