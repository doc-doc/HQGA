import sys
sys.path.insert(0, '../')
import torch
from torch.utils.data import Dataset, DataLoader
from .util import load_file, transform_bb
import os.path as osp
import numpy as np
import nltk
import h5py
import time

class VideoQADataset(Dataset):
    """load the dataset in dataloader"""
    
    def __init__(self, video_feature_path, video_feature_cache, sample_list_path, 
            vocab, multi_choice, use_bert, mode):
        self.video_feature_path = video_feature_path
        self.vocab = vocab
        
        sample_list_file = osp.join(sample_list_path, '{}.csv'.format(mode))
        self.sample_list = load_file(sample_list_file)
        self.video_feature_cache = video_feature_cache
        self.max_qa_length = 20 #20 for MSRVTT, MSVD, TGIF-QA Trans & Action, 37 for nextqa
        self.use_bbox = True
        self.bbox_num = 10 #20 for NExT-QA, 10 for others
        self.use_bert = use_bert
        self.use_frame = True
        self.use_mot = True
        self.multi_choice = multi_choice
        if not self.multi_choice:
            ans_path = osp.join(sample_list_path, 'ans_word.json')
            ans_set = load_file(ans_path)
            self.ans_set = ['<unk>'] + ans_set #index 0 is reserved for out-of-vocab answer
            print('ans size: {}'.format(len(self.ans_set)))

        if self.use_bert:
            bert_path = osp.join(self.video_feature_path, 'qas_bert')
            self.bert_file = osp.join(bert_path, 'bert_ft_{}.h5'.format(mode))

        if self.use_bbox:
            bbox_feat_file = osp.join(self.video_feature_path, 'region_feat_n/region_8c10b_{}.h5'.format(mode))
            print('Load {}...'.format(bbox_feat_file))
                        
            self.bbox_feats = {}
            with h5py.File(bbox_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['feat']
                print(feats.shape) #v_num, clip_num, frame_per_clip, region_per_frame, feat_dim
                bboxes = fp['bbox']
                for id, (vid, feat, bbox) in enumerate(zip(vids, feats, bboxes)):
                    self.bbox_feats[str(vid)] = (feat[:, :, :self.bbox_num, :],bbox[:, :, :self.bbox_num, :]) #(clip, frame, bbox, feat), (clip, frame, bbox, coord)

        if self.use_frame:
            app_feat_file = osp.join(video_feature_path, 'frame_feat/app_feat_{}.h5'.format(mode))
            print('Load {}...'.format(app_feat_file))
            self.frame_feats = {}
            with h5py.File(app_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['resnet_features']
                print(feats.shape) #v_num, clip_num, frame_per_clip, feat_dim
                for id, (vid, feat) in enumerate(zip(vids, feats)):
                    #self.frame_feats[str(vid)] = feat[::2]
                    self.frame_feats[str(vid)] = feat
                
        if self.use_mot:
            mot_feat_file = osp.join(video_feature_path, 'mot_feat/mot_feat_{}.h5'.format(mode))
            print('Load {}...'.format(mot_feat_file))
            self.mot_feats = {}
            with h5py.File(mot_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['resnext_features']
                print(feats.shape) #v_num, clip_num, feat_dim
                for id, (vid, feat) in enumerate(zip(vids, feats)):
                    self.mot_feats[str(vid)] = feat

    def __len__(self):
        return len(self.sample_list)

    def get_video_feature(self, video_name, width=1, height=1):
        """
        :param video_name:
        :param width:
        :param height:
        :return:
        """
        video_feature = []
        if self.use_bbox:
            roi_feat = self.bbox_feats[video_name][0]
            roi_feat = torch.from_numpy(roi_feat).type(torch.float32)
            roi_bbox = self.bbox_feats[video_name][1]

            bbox_feat = transform_bb(roi_bbox, width, height)
            bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)

            region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)
            video_feature.append(region_feat)
            
        if self.use_frame:
            
            temp_feat = self.frame_feats[video_name]
            app_feat = torch.from_numpy(temp_feat).type(torch.float32)
            video_feature.append(app_feat)
           

        if self.use_mot:
            temp_feat = self.mot_feats[video_name]
            mot_feat = torch.from_numpy(temp_feat).type(torch.float32)
            video_feature.append(mot_feat)
            
        return video_feature

    def get_word_idx(self, text, mode='q'):
        """
        convert relation to index sequence
        :param relation:
        :return:
        """
        thd = 25 #13 
        #thd = 13 if mode=='q' else 4 #20 for frameqa #25
        tokens = nltk.tokenize.word_tokenize(str(text).lower())
        token_ids = [self.vocab(token) for i, token in enumerate(tokens) if i < thd]

        return token_ids

    def get_Trans_matrix(self, candidates):
        qa_lengths = [len(qa) for qa in candidates]
        candidates_matrix = torch.zeros([5, self.max_qa_length]).long()
        for k in range(5):
            sentence = candidates[k]
            candidates_matrix[k, :qa_lengths[k]] = torch.Tensor(sentence)

        return candidates_matrix, qa_lengths

    def get_multi_choice_sample(self, idx):
        cur_sample = self.sample_list.loc[idx]
        vid, qid = 'video', 'qid'
        
        video_name, qns, ans, qid = str(cur_sample[vid]), str(cur_sample['question']), \
                                    int(cur_sample['answer']), str(cur_sample[qid])
        width, height = int(cur_sample['width']), int(cur_sample['height'])
        candidate_qas = []
        qns2ids = [self.vocab('<start>')] + self.get_word_idx(qns) + [self.vocab('<end>')]
        for id in range(5):
            cand_ans = cur_sample['a' + str(id)]
            ans2id = self.get_word_idx(cand_ans, 'a') + [self.vocab('<end>')]
            candidate_qas.append(qns2ids + ans2id)

        candidate_qas, qa_lengths = self.get_Trans_matrix(candidate_qas)
        if self.use_bert:
            with h5py.File(self.bert_file, 'r') as fp:
                temp_feat = fp['feat'][idx]
                candidate_qas = torch.from_numpy(temp_feat).type(torch.float32)
            qa_lengths = []
            for i in range(5):
                valid_row = nozero_row(candidate_qas[i])
                assert valid_row != 0, f'{video_name}, {qid}'
                # if valid_row != qa_lengths[i]:
                qa_lengths.append(valid_row)
        qns_key = video_name + '_' + qid
        return video_name, candidate_qas, qa_lengths, ans, qns_key, width, height


    def __getitem__(self, idx):
        """
        return an item from data list as tuple (video, relation)
        :param idx:
        :return:
                -video: torch tensor (nframe, nbbox, feat)
                -relation: torch tensor of variable length
        """
        if self.multi_choice:
            video_name, candidate_qas, qa_lengths, ans_idx, qns_key, width, height = self.get_multi_choice_sample(idx)
        else:
            cur_sample = self.sample_list.loc[idx]
            video_name, qns, ans, qid = str(cur_sample['video']), str(cur_sample['question']), \
                                        str(cur_sample['answer']), str(cur_sample['qid'])
            # width, height = 320, 240 #msrvtt
            width, height = cur_sample['width'], cur_sample['height']
            candidate_qas = self.get_word_idx(qns)
            qa_lengths = len(candidate_qas)
            if self.use_bert:
                with h5py.File(self.bert_file, 'r') as fp:
                    temp_feat = fp['feat'][idx]
                    candidate_qas = torch.from_numpy(temp_feat).type(torch.float32)
                qa_lengths = nozero_row(candidate_qas)
            ans_idx = self.ans_set.index(ans) if ans in self.ans_set else 0
            qns_key = str(qid)

        video_feature = self.get_video_feature(video_name, width, height)
        qa_lengths = torch.tensor(qa_lengths)

        return video_feature, candidate_qas, qa_lengths, ans_idx, qns_key

def nozero_row(A):
    i = 0
    for row in A:
        if row.sum()==0:
            break
        i += 1

    return i


class QALoader():
    def __init__(self, batch_size, num_worker, video_feature_path, video_feature_cache,
                 sample_list_path, vocab, multi_choice, use_bert,train_shuffle=True, val_shuffle=False):
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.video_feature_path = video_feature_path
        self.video_feature_cache = video_feature_cache
        self.sample_list_path = sample_list_path
        self.vocab = vocab
        self.use_bert = use_bert
        self.multi_choice = multi_choice
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle

    def run(self, mode=''):
        if mode != 'train':
            train_loader = ''
            val_loader = self.validate(mode)
        else:
            train_loader = self.train('train')
            val_loader = self.validate('val')
        return train_loader, val_loader

    def train(self, mode):
        # print("Now in train")
        training_set = VideoQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                       self.vocab, self.multi_choice, self.use_bert, mode)

        print('Eligible video-qa pairs for training : {}'.format(len(training_set)))
        if not self.multi_choice and not self.use_bert:
            train_loader = DataLoader(
                dataset=training_set,
                batch_size=self.batch_size,
                shuffle=self.train_shuffle,
                num_workers=self.num_worker,
                collate_fn= collate_fn
                )
        else:
            train_loader = DataLoader(
                dataset=training_set,
                batch_size=self.batch_size,
                shuffle=self.train_shuffle,
                num_workers=self.num_worker,
                )

        return train_loader

    def validate(self, mode):
        # print("Now in Validate")
        # for validation videos
        validation_set = VideoQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                         self.vocab, self.multi_choice, self.use_bert, mode)

        print('Eligible video-qa pairs for validation/test : {}'.format(len(validation_set)))
        if not self.multi_choice and not self.use_bert:
            val_loader = DataLoader(
                dataset=validation_set,
                batch_size=self.batch_size,
                shuffle=self.val_shuffle,
                num_workers=self.num_worker,
                collate_fn=collate_fn
                )
        else:
            val_loader = DataLoader(
                dataset=validation_set,
                batch_size=self.batch_size,
                shuffle=self.val_shuffle,
                num_workers=self.num_worker,
                )
        return val_loader

def collate_fn (data):
    """
    Create mini-batch tensors from the list of tuples (video, qns, qns_len, ans, qid)
    """
    data.sort(key=lambda x : len(x[1]), reverse=True)
    videos, qnss_ori, qns_lens, anss, qids = zip(*data)
    temp_videos = []
    feat_num = len(videos[0]) #([a,b],[a,b],[a,b],[a,b])
    for fid in range(feat_num):
        temp = torch.stack([videos[i][fid] for i in range(len(videos))], 0)
        temp_videos.append(temp)
    videos = temp_videos

    anss = torch.LongTensor(anss)
    qns_lens = torch.LongTensor(qns_lens)
    qnss = torch.zeros(len(qnss_ori), max(qns_lens)).long()
    for i, qns in enumerate(qnss_ori):
        end = qns_lens[i]
        qnss[i, :end] = torch.LongTensor(qns[:end])
    
    return videos, qnss, qns_lens, anss, qids
