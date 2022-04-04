import numpy as np
import torch
import logging
from model import RippleNet
from sklearn.metrics import roc_auc_score, f1_score
from prettytable import PrettyTable
from evaluate import test
from time import time
from utils import get_feed_dict
from helper import early_stopping

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

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    with open(f"./training_log/RippleNet_{args.dataset}_{args.lr}.txt","w") as f:
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

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                                stopping_step, expected_order='acc',
                                                                                flag_step=10)
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                        torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')
            print(result_table)
            f.write(str(result_table)+"\n")

        print('early stopping at %d, recall@20:%.4f' % (step, cur_best_pre_0))
        f.write('early stopping at %d, recall@20:%.4f' % (step, cur_best_pre_0)+"\n")
            

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


