import numpy as np
import torch
import logging
from model import RippleNet
from sklearn.metrics import roc_auc_score, f1_score
from prettytable import PrettyTable
from evaluate import test
from time import time

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
        train_s_t = time()
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size
            if show_loss:
                print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))
        train_e_t = time()
        # evaluation

        test_s_t = time()
        ret = test(args, model, data_info)
        test_e_t = time()
        result_table = PrettyTable()
        result_table.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
        result_table.add_row(
            [step, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
        )

        print(result_table)
        # train_auc, train_acc = evaluation(args, model, train_data, ripple_set, args.batch_size)
        # eval_auc, eval_acc = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        # test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)
        # eval_auc, eval_f1 = ctr_eval(args, model, eval_data, ripple_set, args.batch_size)
        # test_auc, test_f1 = ctr_eval(args, model, test_data, ripple_set, args.batch_size)
        # ctr_info = 'epoch {}    eval auc: {:.4f} f1: {:.4f}    test auc: {:.4f} f1: {:.4f}'.format(step, eval_auc, eval_f1, test_auc, test_f1)
        # print(ctr_info)
        
        # if args.show_topk:
        #     print(topk_eval(args, model, train_data, test_data, ripple_set))
        

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