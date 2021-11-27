from networks import embed_loss
from networks.VQAModel import HQGA
from networks.Encoder import EncoderQns, EncoderVid
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import torch.nn as nn


class VideoQA():
    def __init__(self, vocab, train_loader, val_loader, glove_embed, checkpoint_path, model_type,
                 model_prefix, vis_step, lr_rate, batch_size, epoch_num, grad_accu_steps, use_bert=True, multi_choice=True):
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.glove_embed = glove_embed
        self.model_dir = checkpoint_path
        self.model_type = model_type
        self.model_prefix = model_prefix
        self.vis_step = vis_step
        self.lr_rate = lr_rate
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.use_bert = use_bert
        self.multi_choice = multi_choice
        self.accu_grad_step = grad_accu_steps

    def build_model(self):

        feat_dim = 2048
        bbox_dim = 5
        num_clip, num_frame, num_bbox = 8, 8*4, 10
        feat_hidden, pos_hidden = 256, 128
        word_dim = 300
        vocab_size = None if self.use_bert else len(self.vocab)
        
        num_class = 1 if self.multi_choice else 1853 #4001 for msrvtt, 1853 for msvd, 1541 for frameQA in TGIF-QA

        if self.model_type == 'HQGA':
            
            vid_encoder = EncoderVid.EncoderVid(feat_dim, bbox_dim, num_clip, num_frame, num_bbox,
                                                     feat_hidden, pos_hidden, input_dropout_p=0.3)

            qns_encoder = EncoderQns.EncoderQns(word_dim, feat_hidden, vocab_size, self.glove_embed, use_bert=self.use_bert,
                                                n_layers=1, rnn_dropout_p=0, input_dropout_p=0.3, bidirectional=True,
                                                rnn_cell='gru')

            self.model = HQGA.HQGA(vid_encoder, qns_encoder, self.device, num_class)

        params = [{'params':self.model.parameters()}]
        self.optimizer = torch.optim.Adam(params = params, lr=self.lr_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=5, verbose=True)
        
        self.model.to(self.device)
        if self.multi_choice:
            self.criterion = embed_loss.MultipleChoiceLoss().to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)


    def save_model(self, epoch, acc):
        torch.save(self.model.state_dict(), osp.join(self.model_dir, '{}-{}-{}-{:.2f}.ckpt'
                                                     .format(self.model_type, self.model_prefix, epoch, acc)))

    def resume(self, model_file):
        """
        initialize model with pretrained weights
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        print(f'Warm-starting from model {model_path}')
        model_dict = torch.load(model_path)
        new_model_dict = {}
        for k, v in self.model.state_dict().items():
            if k in model_dict:
                v = model_dict[k]
            else:
                pass
                # print(k)
            new_model_dict[k] = v
        self.model.load_state_dict(new_model_dict)


    def run(self, model_file, pre_trained=False):
        self.build_model()
        best_eval_score = 0.0
        if pre_trained:
            self.resume(model_file)
            best_eval_score = self.eval(0)
            print('Initial Acc {:.2f}'.format(best_eval_score))

        for epoch in range(1, self.epoch_num):
            train_loss, train_acc = self.train(epoch)
            eval_score = self.eval(epoch)
            print("==>Epoch:[{}/{}][Train Loss: {:.4f} Train acc: {:.2f} Val acc: {:.2f}]".
                  format(epoch, self.epoch_num, train_loss, train_acc, eval_score))
            self.scheduler.step(eval_score)
            if eval_score >= best_eval_score:
                best_eval_score = eval_score
                if epoch >= 3 or pre_trained:
                    self.save_model(epoch, best_eval_score)

    def train(self, epoch):
        print('==>Epoch:[{}/{}][lr_rate: {}]'.format(epoch, self.epoch_num, self.optimizer.param_groups[0]['lr']))
        self.model.train()
        total_step = len(self.train_loader)
        epoch_loss = 0.0
        prediction_list = []
        answer_list = []

        for iter, inputs in enumerate(self.train_loader):
            videos, qas, qas_lengths, answers, qns_keys = inputs
            video_inputs = to_device(videos, self.device)
            qas_inputs = qas.to(self.device)
            ans_targets = answers.to(self.device)
            out, prediction, _ = self.model(video_inputs, qas_inputs, qas_lengths)

            loss = self.criterion(out, ans_targets)
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12)
            loss /= self.accu_grad_step
            loss.backward()
            if (iter+1) % self.accu_grad_step == 0 or (iter == total_step):
                self.optimizer.step()
                self.model.zero_grad()
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if iter % (self.vis_step * self.accu_grad_step) == 0:
                print('\t[{}/{}]-{}-{:.4f}'.format(iter, total_step,cur_time, loss.item()*self.accu_grad_step))
            epoch_loss += loss.item() * self.accu_grad_step

            prediction_list.append(prediction)
            answer_list.append(answers)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers==ref_answers).numpy()
        if not self.multi_choice:
            acc_num -= unk_num(predict_answers, ref_answers)
        return epoch_loss / total_step, acc_num*100.0 / len(ref_answers)

    def eval(self, epoch):
        print('==>Epoch:[{}/{}][validation stage]'.format(epoch, self.epoch_num))
        self.model.eval()
        prediction_list = []
        answer_list = []
        with torch.no_grad():
            for iter, inputs in enumerate(self.val_loader):
                videos, qas, qas_lengths, answers, qns_keys = inputs
                video_inputs = to_device(videos, self.device)
                qas_inputs = qas.to(self.device)
                out, prediction, _ = self.model(video_inputs, qas_inputs, qas_lengths)

                prediction_list.append(prediction)
                answer_list.append(answers)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers == ref_answers).numpy()
        if not self.multi_choice:
            acc_num -= unk_num(predict_answers, ref_answers)

        return acc_num*100.0 / len(ref_answers)


    def predict(self, model_file, result_file):
        """
        predict the answer with the trained model
        :param model_file:
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        self.build_model()
        if self.model_type in['HGA', 'STVQA', 'HCG']:
            self.resume(model_file)
        else:
            old_state_dict = torch.load(model_path)
            self.model.load_state_dict(old_state_dict)

        self.model.eval()
        results = {}
        with torch.no_grad():
            for it, inputs in enumerate(self.val_loader):
                
                videos, qas, qas_lengths, answers, qns_keys = inputs
                
                video_inputs = to_device(videos, self.device)
                qas_inputs = qas.to(self.device)
                out, prediction, vis_graph = self.model(video_inputs, qas_inputs, qas_lengths)
                prediction = prediction.data.cpu().numpy()
                answers = answers.numpy()
                # with open('vis/nextqa/{}.pkl'.format(str(qns_keys[0])), 'wb') as fp:
                #         gdata = {}
                #         for k, dic in vis_graph.items():
                #             gdata[k] = {}
                #             for sk, v in dic.items():
                #                 gdata[k][sk] = v.data.cpu().numpy()
                #         pkl.dump(gdata, fp)
                
                for qid, pred, ans in zip(qns_keys, prediction, answers):
                    results[qid] = {'prediction': int(pred), 'answer': int(ans)}
                

        print(len(results))
        save_file(results, result_file)

def unk_num(predictions, references):
    num = predictions.shape[0]
    uk_num = 0
    for i in range(num):
        if predictions[i] == references[i] and references[i] == 0:
            uk_num += 1
    return uk_num