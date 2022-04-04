import torch
import numpy as np

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