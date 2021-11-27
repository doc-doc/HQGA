import sys
sys.path.insert(0, 'networks')
import torch.nn as nn
from PosEmbed import positionalencoding1d
from lgcn import GGCN
import torch
class EncoderVidLGCN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (float): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderVidLGCN, self).__init__()
        self.dim_feat = 2048
        self.dim_bbox = dim_vid - 2048
        self.num_bbox = 10
        self.num_frames = 8 * 4
        self.dim_hidden = dim_hidden

        dim_pos = 128
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.use_bboxPos = True
        self.use_framePos = True
        input_dim = self.dim_feat

        if self.use_bboxPos:
            input_dim += dim_pos
            self.bbox_conv = nn.Sequential(
                nn.Conv2d(self.dim_bbox, dim_pos//2, kernel_size=1),
                nn.BatchNorm2d(dim_pos//2),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Conv2d(64, dim_pos, kernel_size=1),
                nn.BatchNorm2d(dim_pos),
                nn.ReLU(),
                # nn.Dropout(0.5)
            )

        if self.use_framePos:
            input_dim += dim_pos
            self.framePos = positionalencoding1d(dim_pos, self.num_frames) #(fnum, pos_dim)
            self.framePos = self.framePos.unsqueeze(1).expand(-1, self.num_bbox, -1).cuda() #(fnum, num_bbox_perf, pos_dim)

        self.tohid = nn.Sequential(
            nn.Linear(input_dim, dim_hidden*2),
            nn.ELU(inplace=True))

        self.app_conv = nn.Sequential(
            nn.Conv1d(self.dim_feat, dim_hidden*2, 3, padding=1),
            nn.ELU(inplace=True)
        )

        self.gcn = GGCN(
            dim_hidden*2,
            dim_hidden*2,
            dim_hidden*2,
            dropout=input_dropout_p,
            mode=['GCN_sim'],
            skip=True,
            num_layers=2
        )

        # self.num_streams = sum([self.use_image, self.use_bbox, self.use_c3d])
        self.merge = nn.Sequential(
            nn.Linear(self.dim_hidden * 4, self.dim_hidden*2),
            nn.ELU(inplace=True)
        )


    def forward(self, vid_feats):

        roi_feat = vid_feats[0][:,:,:,:,:self.dim_feat]
        batch_size, num_clip, frame_pclip, region_pframe, _ = roi_feat.shape
        roi_feat = roi_feat.view(batch_size, num_clip*frame_pclip, region_pframe, -1)
        # roi_feat = vid_feats[:, :, :, :self.dim_feat]
        bbox_features = roi_feat
        roi_bbox = vid_feats[0][:, :, :, :, self.dim_feat:(self.dim_feat + self.dim_bbox)]
        roi_bbox = roi_bbox.view(batch_size, num_clip*frame_pclip, region_pframe, -1)

        """bboxPos and framePos"""
        if self.use_bboxPos:
            bbox_pos = self.bbox_conv(roi_bbox.permute(
                0, 3, 1, 2)).permute(0, 2, 3, 1)
            bbox_features = torch.cat([roi_feat, bbox_pos], dim=-1)

        if self.use_framePos:
            framePos = self.framePos.unsqueeze(0).expand(batch_size, -1, -1, -1)
            bbox_features = torch.cat([bbox_features, framePos], dim=-1)

        bbox_features = bbox_features.view(batch_size*num_clip*frame_pclip*region_pframe, -1)
        bbox_features = self.tohid(bbox_features)
        bbox_features = bbox_features.view(batch_size, num_clip*frame_pclip*region_pframe, -1)

        video_length = torch.tensor([num_clip * frame_pclip * region_pframe] * batch_size, dtype=torch.long)
        roi_feature = self.gcn(bbox_features, video_length)

        app_feat = vid_feats[1]

        app_feat = app_feat.view(batch_size, -1, self.dim_feat)
        app_feat = self.app_conv(app_feat.transpose(1,2)).transpose(1,2)
        app_feat = app_feat.unsqueeze(2).expand(-1, -1, region_pframe, -1)
        app_feat = app_feat.reshape(batch_size, num_clip*frame_pclip*region_pframe, -1)

        streams = [app_feat, roi_feature]
        streams = torch.cat(streams, dim=-1)

        video_embedding = self.merge(streams)
        return video_embedding, video_length
