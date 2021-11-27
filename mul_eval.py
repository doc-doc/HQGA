import os.path as osp
from utils import load_file


def classified_metric(sample_list_file, result_file):

    sample_list = load_file(sample_list_file)
    group = {'EW':[], 'EH':[], 'TN':[], 'TC':[], 'CC':[], 'CL':[], 'CO':[]}
    for id, row in sample_list.iterrows():
        qns_id = str(row['video']) + '_' + str(row['qid'])
        qtype = str(row['type'])
        #merge predictive qns (previous and next)
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)

    preds = load_file(result_file)
    group_acc = {'EW': 0, 'EH': 0, 'TN': 0, 'TC': 0, 'CC': 0, 'CL': 0, 'CO': 0}
    group_cnt = {'EW': 0, 'EH': 0, 'TN': 0, 'TC': 0, 'CC': 0, 'CL': 0, 'CO': 0}
    overall_acc = {'E':0, 'T':0, 'C':0}
    overall_cnt = {'E':0, 'T':0, 'C':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0 #len(qns_ids)
        acc = 0
        for qid in qns_ids:
            # if qid not in preds: continue
            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']
            #print(type(answer), type(pred))
            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt


    #group_cnt = {}
    #for qtype in group_acc:
    #    group_cnt[qtype] = len(group[qtype])
    print('')
    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    for qtype in group_acc:
        print(qtype, end='\t')
    print('')
    for qtype, acc in group_acc.items():
        print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))


def main(result_file, dataset_dir, mode='val'):
    #dataset_dir = 'dataset/tgifqa/frameqa/'
    data_set = mode
    sample_list_file = osp.join(dataset_dir, data_set+'.csv')
    print('Evaluating {}'.format(result_file))

    classified_metric(sample_list_file, result_file)



if __name__ == "__main__":
    dataset_dir = 'dataset/nextqa/'
    model_type = 'HGA'
    mode = 'val'
    model_prefix = 'bert-ft-h256-{}'.format(32)
    result_file = 'results/{}-{}.json'.format(model_type, model_prefix)
    # result_file = 'results/heuristic/pred_long.json'

    # result_file = '../hcrn-videoqa/results/expVidQA16/preds/{}_preds.json'.format(mode)
    # result_file = 'results/user_performance.json'
    main(result_file, dataset_dir, mode)
