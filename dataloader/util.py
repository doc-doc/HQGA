import json
import os
import os.path as osp
import numpy as np
import pickle as pkl
import pandas as pd

def load_file(file_name):
    annos = None
    if osp.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos

def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = osp.dirname(filename)
    if filepath != '' and not osp.exists(filepath):
        os.makedirs(filepath)
    else:
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)

def pkload(file):
    data = None
    if osp.exists(file) and osp.getsize(file) > 0:
        with open(file, 'rb') as fp:
            data = pkl.load(fp)
        # print('{} does not exist'.format(file))
    return data


def pkdump(data, file):
    dirname = osp.dirname(file)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    with open(file, 'wb') as fp:
        pkl.dump(data, fp)


def transform_bb(roi_bbox, width, height):
    dshape = list(roi_bbox.shape)
    tmp_bbox = roi_bbox.reshape([-1, 4])
    relative_bbox = tmp_bbox / np.asarray([width, height, width, height])
    relative_area = (tmp_bbox[:, 2] - tmp_bbox[:, 0] + 1) * \
                    (tmp_bbox[:, 3] - tmp_bbox[:, 1] + 1)/ (width*height)
    relative_area = relative_area.reshape(-1, 1)
    bbox_feat = np.hstack((relative_bbox, relative_area))
    dshape[-1] += 1
    bbox_feat = bbox_feat.reshape(dshape)

    return bbox_feat


def select_feature(video_feature_path, video_name, video_feature_cache, 
                    frame_path, num_clip=16, frame_pclip=16, bbox_num=10):

    nframe_ds = len(os.listdir(osp.join(frame_path, video_name)))
    clips = sample_clips(nframe_ds, num_clip, frame_pclip)
    video_feat_dir = osp.join(video_feature_path, video_name)
    frame_feat_files = sorted(os.listdir(video_feat_dir))
    frame_dict = {}
    
    for frame_feat_file in frame_feat_files:
        try:
            RoIInfo = np.load(osp.join(video_feat_dir, frame_feat_file))
        except:
            return video_name+' '+frame_feat_file.split('.')[0]
        fid = frame_feat_file.split('.')[0]
        bbox_feat, bbox_coord = RoIInfo['x'], RoIInfo['bbox']
        bnum = bbox_feat.shape[0]
        if bnum < bbox_num:
            add_num = bbox_num - bnum
            # print(add_num)
            add_feat, add_bbox = [], []
            for _ in range(add_num):
                add_feat.append(bbox_feat[-1])
                add_bbox.append(bbox_coord[-1])
            add_feat = np.asarray(add_feat)
            add_bbox = np.asarray(add_bbox)
            # print(add_feat.shape, add_bbox.shape)
            bbox_feat = np.concatenate((bbox_feat, add_feat), axis=0)
            bbox_coord = np.concatenate((bbox_coord, add_bbox), axis=0)
        frame_dict[fid] = {'bbox': bbox_coord[:bbox_num], 'feat': bbox_feat[:bbox_num]}  # (top 20 bbox & feat)
    video_feat = []
    video_bbox = []
    for clip in clips:
        clip_feat = []
        clip_bbox = []
        for fid in clip:
            if fid not in frame_dict:
                feedback = video_name+' '+fid
                return feedback
            clip_feat.append(frame_dict[fid]['feat'])
            clip_bbox.append(frame_dict[fid]['bbox'])
        video_feat.append(clip_feat)
        video_bbox.append(clip_bbox)
    video_feat = np.asarray(video_feat, dtype=np.float32)
    video_bbox = np.asarray(video_bbox, dtype=np.float32)
    # print(video_feat.shape, video_bbox.shape) #(16,4,20,2048), (16,4,20,4)
    cache_file = osp.join(video_feature_cache, video_name)
    np.savez_compressed(cache_file, feat=video_feat, bbox=video_bbox)
    return video_feat, video_bbox

def sample_clips(total_frames, num_clips, num_frames_per_clip):
    clips = []
    frames = [str(f+1).zfill(6) for f in range(total_frames)]
    vis = 0
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1: num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        clip_start = 0 if clip_start < 0 else clip_start
        clip_end = total_frames if clip_end > total_frames else clip_end
        clip = frames[clip_start:clip_end] # evenly sample 4 frames from clip of size 16
        if clip_start == 0 and len(clip) < num_frames_per_clip:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_fids = []
            for _ in range(shortage):
                added_fids.append(frames[clip_start])
            if len(added_fids) > 0:
                clip = added_fids + clip
        if clip_end == total_frames and len(clip) < num_frames_per_clip:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_fids = []
            for _ in range(shortage):
                added_fids.append(frames[clip_end-1])
            if len(added_fids) > 0:
                clip += added_fids
        clip = clip[::4]
        clips.append(clip)

    return clips
