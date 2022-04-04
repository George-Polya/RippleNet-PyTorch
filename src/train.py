import numpy as np
import torch
import logging
from model import RippleNet
from sklearn.metrics import roc_auc_score, f1_score


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    model = RippleNet(args, n_entity, n_relation)
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )

    for step in range(args.n_epoch):
        # training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size
            if show_loss:
                print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))

        # evaluation
        # train_auc, train_acc = evaluation(args, model, train_data, ripple_set, args.batch_size)
        # eval_auc, eval_acc = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        # test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)
        eval_auc, eval_f1 = ctr_eval(args, model, eval_data, ripple_set, args.batch_size)
        test_auc, test_f1 = ctr_eval(args, model, test_data, ripple_set, args.batch_size)
        ctr_info = 'epoch {}    eval auc: {:.4f} f1: {:.4f}    test auc: {:.4f} f1: {:.4f}'.format(step, eval_auc, eval_f1, test_auc, test_f1)
        print(ctr_info)
        
        if args.show_topk:
            print(topk_eval(args, model, train_data, test_data, ripple_set))
        

def ctr_eval(args, model, data, ripple_set, batch_size):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:
        labels = data[start:start + args.batch_size, 2]
        return_dict = model(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        scores = return_dict["scores"]
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()  
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1


def topk_eval(args, model, train_data, test_data, ripple_set):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}

    item_set = set(train_data[:,1].tolist() + test_data[:,1].tolist())
    train_record = _get_user_record(args, train_data, True)
    test_record = _get_user_record(args, test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 12855
    if len(user_list) > user_num:
        np.random.seed()    
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    data = np.vstack([train_data, test_data])
    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size] 
            # input_data = _get_topk_feed_data(user, items)
            return_dict = model(*get_feed_dict(args, model, data, ripple_set, start, start + args.batch_size))
            scores = return_dict["scores"]
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            # input_data = _get_topk_feed_data(user, res_items)
            return_dict = model(*get_feed_dict(args, model, data, ripple_set, start, start + args.batch_size))
            scores = return_dict["scores"]
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
    model.train()  
    recall = [np.mean(recall_list[k]) for k in k_list]
    print(recall)
    res = ""
    for i,j in zip(k_list, recall):
        res += "K@%d:%.4f  "%(i,j)
    return res
    

def get_feed_dict(args, model, data, ripple_set, start, end):
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], []
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    if args.use_cuda:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return items, labels, memories_h, memories_r,memories_t


def evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    model.eval()
    while start < data.shape[0]:
        auc, acc = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list))



def topk_eval(args, model, train_data, test_data, ripple_set):
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k:[] for k in k_list}

    item_set = set(train_data[:,1].tolist() + test_data[:,1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & (test_record.keys()) )
    user_num = 13498
    if len(user_list) > user_num:
        np.random.seed(2022)
        user_list = np.random.choice(user_list,size=user_num, replace=False)
    data = np.vstack([train_data, test_data])
    model.eval()
    for user in user_list:
        test_item_list = list(item_set - set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start+args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            scores = model(*get_feed_dict(args, model, data, ripple_set, start, start + args.batch_size))
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size

        if start < len(test_item_list):
            res_items = test_item_list[start:] +[test_item_list[-1]]*(args.batch_size - len(test_item_list)+start)
            input_data = _get_topk_feed_data(user, items)
            scores = model(*get_feed_dict(args, model, data, ripple_set, start, start + args.batch_size))
            for item, score in zip(items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x : x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
        model.train()
        recall = [np.mean(recall_list[k]) for k in k_list]
        _show_recall_info(zip(k_list, recall))


def _show_recall_info(recall_zip):
    res = ""
    for i,j in recall_zip:
        res += "K@%d:%.4f  "%(i,j)
    # logging.info(res)
    return res



def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user,item])
    return np.array(res)

def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
                # user_history_dict[user] = list()
            user_history_dict[user].add(item)
            # user_history_dict[user].append(item)
    return user_history_dict

def _get_item_record(data):
    item_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]

        if label == 1:
            if item not in item_dict:
                item_dict[item] = set()
            item_dict[item].add(user)

    return item_dict