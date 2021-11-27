import sys
sys.path.insert(0, 'networks')
import torch.nn as nn
from PosEmbed import positionalencoding1d
import torch

class EncoderVid(nn.Module):
    def __init__(self, feat_dim, bbox_dim, num_clip, num_frame, num_bbox, feat_hidden, pos_hidden, input_dropout_p=0.3):
        
        super(EncoderVid, self).__init__()
        self.dim_feat = feat_dim
        self.dim_bbox = bbox_dim
        self.num_clip = num_clip
        self.num_bbox = num_bbox
        self.num_frame = num_frame
        self.dim_hidden = feat_hidden * 2
        self.input_dropout_p = input_dropout_p

        self.use_bboxPos = True
        self.use_framePos = True
        input_dim = feat_dim

        if self.use_bboxPos:
            input_dim += pos_hidden
            self.bbox_conv = nn.Sequential(
                nn.Conv2d(self.dim_bbox, pos_hidden//2, kernel_size=1),
                nn.BatchNorm2d(pos_hidden//2),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Conv2d(pos_hidden//2, pos_hidden, kernel_size=1),
                nn.BatchNorm2d(pos_hidden),
                nn.ReLU(),
                # nn.Dropout(0.5)
            )

        if self.use_framePos:
            input_dim += pos_hidden
            self.framePos = positionalencoding1d(pos_hidden, self.num_frame).cuda() #(fnum, pos_dim)
            self.framePos = self.framePos.unsqueeze(1).expand(-1, self.num_bbox, -1) #(fnum, num_bbox_perf, pos_dim)

        self.tohid = nn.Sequential(
            nn.Linear(input_dim, self.dim_hidden),
            nn.ELU(inplace=True))
        
        # self.merge_fr = nn.Sequential(
        #     nn.Linear(self.feat_hidden*2, self.feat_hidden),
        #     nn.ELU(inplace=True))

        self.app_conv = nn.Sequential(
            nn.Conv1d(feat_dim, self.dim_hidden, 3, padding=1),
            nn.ELU(inplace=True)
        )

        self.mot_conv = nn.Sequential(
            nn.Conv1d(feat_dim, self.dim_hidden, 3, padding=1),
            nn.ELU(inplace=True)
        )

    def forward(self, vid_feats):
         
        roi_feat = vid_feats[0][:,:,:,:,:self.dim_feat]
        batch_size, num_clip, frame_pclip, region_pframe, dim_feat = roi_feat.size()
        roi_bbox = vid_feats[0][:, :, :, :, dim_feat:(dim_feat+self.dim_bbox)].\
            view([batch_size,num_clip*frame_pclip,region_pframe,-1])
        bbox_features = roi_feat
        
        if self.use_bboxPos:
            bbox_pos = self.bbox_conv(roi_bbox.permute(
                0, 3, 1, 2)).permute(0, 2, 3, 1)
            bbox_pos = bbox_pos.view([batch_size, num_clip, frame_pclip, region_pframe,-1])
            bbox_features = torch.cat([roi_feat, bbox_pos], dim=-1)

        if self.use_framePos:
            framePos = self.framePos.unsqueeze(0).expand(batch_size, -1, -1, -1)
            framePos = framePos.view(batch_size, num_clip, frame_pclip, region_pframe, -1)
            bbox_features = torch.cat([bbox_features, framePos], dim=-1)
        
        bbox_features = self.tohid(bbox_features)
        bbox_feat = bbox_features
        
        app_feat = vid_feats[-2]
        batch_size, num_clip, frame_pclip, dim = app_feat.shape
        app_feat = app_feat.reshape(batch_size, -1, dim)
        app_feat = self.app_conv(app_feat.transpose(1, 2)).transpose(1, 2)
        app_feat = app_feat.view(batch_size, num_clip, frame_pclip, -1)
        
        mot_feat = vid_feats[-1]
        mot_feat = self.mot_conv(mot_feat.transpose(1, 2)).transpose(1, 2)
                
        return (bbox_feat, app_feat, mot_feat)