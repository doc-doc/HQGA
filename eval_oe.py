from utils import *
from nltk.corpus import wordnet as wn

qns_type_total = {'what':0, 'who':0, 'how':0, 'when':0, 'where':0}
qns_type_right = {'what':0, 'who':0, 'how':0, 'when':0, 'where':0}

what_type_total = {'a':0, 'o':0}
what_type_right = {'a':0, 'o':0}

def classified_metric(gt_file, pred_file):
    samples = pd.read_csv(gt_file)
    sp_num = len(samples)
    qid2type = {}
    qid2tag = {}
    for idx, row in samples.iterrows():
        qid, qns = str(row['qid']), str(row['question'])
        s_qns = qns.split(' ')
        qtype = s_qns[0]
        qid2type[qid] = qtype
        qns_type_total[qtype] += 1
        #############################################
        if qtype == 'what':
            tag = 'o'
            if 'doing' in s_qns: tag = 'a'
            qid2tag[qid] = tag
            what_type_total[tag] += 1
        #############################################

    res = load_file(pred_file)
    total_res = len(res)
    assert sp_num == total_res, 'incomplete prediction'
    cnt = 0
    for qid, value in res.items():
        # qid = str(int(qid)+170859) #map to hcrn id
        #skip unknown
        if value['prediction'] == 0: 
            continue
        if value['prediction'] == value['answer']:
            cnt += 1
            qns_type_right[qid2type[qid]] += 1
            #############################################
            if qid2type[qid] == 'what':
                what_type_right[qid2tag[qid]] += 1
            #############################################


    for qtype, value in qns_type_right.items():
        qns_type_right[qtype] = value*100/qns_type_total[qtype]
        print(qtype, end='\t')
    print('All')
    for qtype, value in qns_type_right.items():
        print('{:.2f}'.format(value), end='\t')

    print('{:.2f}'.format(cnt*100/sp_num))

    ###################################
    print(what_type_total)
    for tag in what_type_right:
        print('what_'+tag, end='\t')
    print('')
    for tag, cnt in what_type_right.items():
        print('{:.2f}'.format(cnt*100.0/what_type_total[tag]), end='\t')
    print('')
    for tag, cnt in what_type_right.items():
        print('{:.2f}'.format(what_type_total[tag]/qns_type_total['what']), end='\t')
    print('')


def overall_acc(sample_list_file, result_file):
    samples = load_file(sample_list_file)
    qids = list(samples['qid'])
    total_num = len(qids)
    predictions = load_file(result_file)
    acc_num = 0
    print(total_num, len(predictions))
    for qid in qids:
        if qid not in predictions: continue
        if predictions[qid]['prediction'] == predictions[qid]['answer']:
            acc_num += 1
    print('{:.2f}'.format(acc_num*100.0/total_num))



def main(result_file, dataset_dir, mode='val'):
    
    data_set = mode
    sample_list_file = osp.join(dataset_dir, data_set+'.csv')
    print('Evaluating {}'.format(result_file))
    if 'msrvtt' in dataset_dir.split('/') or 'msvd' in dataset_dir.split('/'):
        classified_metric(sample_list_file, result_file)
    else:
        overall_acc(sample_list_file, result_file)


if __name__ == "__main__":
    model_type = 'HQGA'
    mode = 'test'
    dataset = 'msvd'
    task = ''
    model_prefix = 'bert-8c10b-2L05GCN-FCV-AC-VM-{}'.format(mode)
    result_file = 'results/{}/{}/{}-{}.json'.format(dataset,task, model_type, model_prefix)
    dataset_dir = 'dataset/{}/{}'.format(dataset, task)
    main(result_file, dataset_dir, mode)
