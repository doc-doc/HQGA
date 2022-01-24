import sys
sys.path.insert(0 , 'networks')
import torch
import torch.nn as nn
import random as rd
import numpy as np
from biatt import BiAttn
from graph import GCN

class HQGA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device, num_class):
        """
        Video as Conditional Graph Hierarchy for Multi-Granular Question Answering
        """
        super(HQGA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.num_class = num_class
        self.device = device
        hidden_size = vid_encoder.dim_hidden
        input_dropout_p = vid_encoder.input_dropout_p
        layer_num = 2
        half = 2

        self.bidirec_att = BiAttn(None, 'dot', get_h=False)
        self.gcn_region = GCN(
            hidden_size,
            hidden_size//half,
            hidden_size,
            dropout=input_dropout_p,
            skip=True,
            num_layers=layer_num
        )
        if num_class >= 1:
            self.gcn_atten_pool_region = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
                nn.Softmax(dim=-2))
    
        self.merge_rf = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ELU(inplace=True)
        )
        
        self.gcn_frame = GCN(
            hidden_size,
            hidden_size//half,
            hidden_size,
            dropout=input_dropout_p,
            skip=True,
            num_layers=layer_num
        )

        if num_class >= 1:
            self.gcn_atten_pool_frame = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
                nn.Softmax(dim=-2))

        self.merge_cm = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ELU(inplace=True)
        )

        self.gcn_clip = GCN(
            hidden_size,
            hidden_size//half,
            hidden_size,
            dropout=input_dropout_p,
            skip=True,
            num_layers=layer_num
        )
        if num_class >= 1:
            self.gcn_atten_pool_clip = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
                nn.Softmax(dim=-2))
        
        if num_class == 1:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, num_class)
            )
        else:
            self.classifier = nn.Sequential(nn.Dropout(0.15),
                                nn.Linear(hidden_size * 2, hidden_size),
                                nn.ELU(),
                                nn.BatchNorm1d(hidden_size),
                                nn.Dropout(0.15),
                                nn.Linear(hidden_size, num_class))
        
    def forward(self, vid_feats, qas, qas_lengths):
        """
        :param vid_feats:
        :param qns:
        :param qns_lengths:
        :param mode:
        :return:
        """
        vid_feats = self.vid_encoder(vid_feats)
        out = []

        if self.num_class == 1:
            # for multi-choice
            if self.qns_encoder.use_bert:
                batch_size, choice_size, max_len, feat_dim = qas.size()
                cand_qas = qas.view(batch_size*choice_size, max_len, feat_dim)  
                cand_len = qas_lengths.view(batch_size*choice_size)
            else:
                batch_size, choice_size, max_len = qas.size()
                cand_qas = qas.view(batch_size*choice_size, max_len)
                cand_len = qas_lengths.view(batch_size*choice_size)
        else:
            #for open-ended
            batch_size, choice_size = qas.shape[0], 1
            cand_qas = qas
            cand_len = qas_lengths

        q_output, s_hidden = self.qns_encoder(cand_qas, cand_len)
        qns_global = s_hidden.permute(1, 0, 2).contiguous().view(q_output.shape[0], -1)
        # print(q_output.shape, qns_global.shape)
        out, vis_graph = self.vq_encoder(vid_feats, q_output, cand_len, qns_global)
        
        _, predict_idx = torch.max(out, 1)
        return out, predict_idx, vis_graph


    def vq_encoder(self, vid_feats, q_output, cand_len, qns_global):
        """
        :param vid_feats:
        :param qas:
        :param qas_lengths:
        :return:
        """
        hierarchy_out, hierarchy_out_att_pool, num_clips, vis_graph = self.hierarchy(vid_feats, q_output, cand_len)
        if self.num_class == 1:
            #for multi-choice QA
            out = self.classifier(qns_global*hierarchy_out_att_pool).squeeze()
            out = out.view(-1, 5)
        else:
            cb_feat = torch.cat([hierarchy_out_att_pool, qns_global], -1)
            out = self.classifier(cb_feat)
            if cb_feat.shape[0] > 1:
                out = out.squeeze()

        return out, vis_graph

    
    def hierarchy(self, vid_feats, qas_feat, qas_lengths):
        
        #############Go########################
        bbox_feats, app_feats, mot_feats = vid_feats
        if self.num_class == 1:
            batch_size = bbox_feats.shape[0]
            batch_repeat = np.reshape(np.tile(np.expand_dims(np.arange(batch_size),
                                                         axis=1),[1, 5]), [-1])
            bbox_feats, app_feats, mot_feats = bbox_feats[batch_repeat], \
                                                app_feats[batch_repeat], \
                                                mot_feats[batch_repeat]

        batch_size, num_clip, frame_pclip, region_pframe, feat_dim = bbox_feats.size()
        
        xlen = num_clip*frame_pclip*region_pframe
        X = bbox_feats.view(batch_size, xlen, feat_dim)
        X_len = torch.tensor([xlen] * batch_size, dtype=torch.long)
        v_output, QO = self.bidirec_att(X, X_len, qas_feat, qas_lengths)
        v_output += X

        v_output = v_output.view(-1, region_pframe, feat_dim)
        num_rpframe = torch.tensor([region_pframe] * v_output.shape[0], dtype=torch.long)
        
        
        gcn_output_region, GO = self.gcn_region(v_output, num_rpframe)
        if self.num_class >= 1:
            att_region = self.gcn_atten_pool_region(gcn_output_region)
            gcn_att_pool_region = torch.sum(gcn_output_region*att_region, dim=1)
        else:
            gcn_att_pool_region = torch.sum(gcn_output_region, dim=1)

        #############GF########################
        gcn_region_output = gcn_att_pool_region.view(batch_size, num_clip*frame_pclip, -1)
        app_feats_f = app_feats.reshape(batch_size, num_clip*frame_pclip, -1)
        tmp = torch.cat((app_feats_f, gcn_region_output), -1)
        gcn_frame_input = self.merge_rf(tmp)
        
        xlen = num_clip*frame_pclip
        X = gcn_frame_input
        X_len = torch.tensor([xlen] * batch_size, dtype=torch.long)
        v_output, QF = self.bidirec_att(X, X_len, qas_feat, qas_lengths)
        v_output += X
        
        v_output = v_output.view(-1, frame_pclip, feat_dim)
        num_fpclip = torch.tensor([frame_pclip] * v_output.shape[0], dtype=torch.long)
        gcn_output_frame, GF = self.gcn_frame(v_output, num_fpclip)

        if self.num_class >= 1:
            att_frame = self.gcn_atten_pool_frame(gcn_output_frame)
            gcn_att_pool_frame = torch.sum(gcn_output_frame * att_frame, dim=1)
        else:
            gcn_att_pool_frame = torch.sum(gcn_output_frame, dim=1)

        #############GC########################
        gcn_frame_output = gcn_att_pool_frame.view(batch_size, num_clip, -1) 
        
        batch_size, num_clip, _ = mot_feats.size()
        num_clips = torch.tensor([num_clip] * batch_size, dtype=torch.long)
        tmp = torch.cat((gcn_frame_output, mot_feats), -1)
        gcn_clip_input = self.merge_cm(tmp)
        
        
        v_output, QC = self.bidirec_att(gcn_clip_input, num_clips, qas_feat, qas_lengths)
        v_output += gcn_clip_input
        
        gcn_output_clip, GC = self.gcn_clip(v_output, num_clips)
        
        if self.num_class >= 1:
            att_clip = self.gcn_atten_pool_clip(gcn_output_clip)
            gcn_att_pool_clip = torch.sum(gcn_output_clip * att_clip, dim=1)
        else:
            gcn_att_pool_clip = torch.sum(gcn_output_clip, dim=1)
        
        vis_graph = None

        return gcn_output_clip, gcn_att_pool_clip, num_clips, vis_graph
