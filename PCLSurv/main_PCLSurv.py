import sys
import time
import numpy as np
import torch
import torch.nn as nn
import random
import math
import pandas as pd
from sklearn.model_selection import KFold


sys.path.append("..")
from valfunc import AUC, CIndex, run_kmeans
from dataload import CanDataset
from network import PCLSurv
from torch.utils.data import DataLoader

gpu_id = 3
cuda_id = "cuda:"+str(gpu_id)
device = torch.device(cuda_id)


def compute_features(RNASeq, miRNA, model):
    print('Computing features...')
    model.eval()
    with torch.no_grad():
        features = model(RNASeq, miRNA, is_eval=True)
        features = features.squeeze(dim=1)
    return features.cpu()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


result_path = '/home/lcheng/LiZhiMin/PCL-cancer/results/'


def main():
    #seed = 666
    #setup_seed(seed)

    for CancerName in ['aml', 'breast', 'colon', 'gbm', 'kidney', 'liver', 'lung', 'melanoma', 'ovarian', 'sarcoma', 'filtered']:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        RNASeq_dataframe = pd.read_csv('/home/lcheng/LiZhiMin/cancerdata/' + CancerName + '/' + CancerName + 'exp.csv')
        RNASeq_dataframe = RNASeq_dataframe.iloc[:, 1:] 
        miRNA_dataframe = pd.read_csv('/home/lcheng/LiZhiMin/cancerdata/' + CancerName + '/' + CancerName + 'mirna.csv')
        miRNA_dataframe = miRNA_dataframe.iloc[:, 1:]
        clinical_dataframe = pd.read_csv('/home/lcheng/LiZhiMin/cancerdata/' + CancerName + '/' + CancerName + '_survival.csv')
        RNASeq_feature = np.array(RNASeq_dataframe)
        #ind_range = range(RNASeq_feature.shape[0])
        indices = np.arange(RNASeq_feature.shape[0])

        for fold, (train_indices, val_indices) in enumerate(kfold.split(indices), 1):

            data_train = CanDataset(RNASeq_dataframe, miRNA_dataframe, clinical_dataframe, indices[train_indices], gpu_id)
            data_eval = CanDataset(RNASeq_dataframe, miRNA_dataframe, clinical_dataframe, indices[val_indices],gpu_id)
            
            batch_size = 64
            
            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False)
            eval_loader = DataLoader(data_eval, batch_size=data_eval.miRNA_tensor.shape[0], shuffle=True,
                                     drop_last=True)

            model = PCLSurv(data_eval.RNASeq_tensor.shape[1], data_eval.miRNA_tensor.shape[1])
            model.to(device)
            
            optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=2e-3, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss().cuda(gpu_id)

            epochs = 200
            c_index_max = 0

            for epoch in range(epochs):
                
                train(train_loader, model, criterion, optimizer, epoch, CancerName, fold)

                c_index, auc_val = eval_model(eval_loader, model)
                if  c_index > c_index_max:
                    c_index_max = c_index
                    sava_path = result_path + str(fold) + CancerName + 'checkpoint.pth.tar'
                    torch.save(model.state_dict(), sava_path)
                    print('save model')
                    with open(result_path + str(fold) + CancerName + 'Val' + '.log', 'a') as f:
                        f.writelines(
                            'epoch: {} cind:{:.4f} AUC: {:.4f}\n'.format(epoch, c_index, auc_val)
                        )



def train(train_loader, model, criterion, optimizer, epoch, CancerName, seed):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()

    for batch_idx, (RNASeq, miRNA, ystatus, ytime) in enumerate(train_loader):

        features = compute_features(RNASeq, miRNA, model)  # con feature
        number_cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        features = features.numpy()
        temperature = 0.2
        cluster_result = run_kmeans0(features, number_cluster, temperature, gpu_id)

        # compute output
        index = np.arange(0, RNASeq.shape[0])

        rRNA, rmiRNA, output, target, output_proto, target_proto = model(RNASeq, miRNA, cluster_result=cluster_result, index=index, gpu_id=gpu_id)

        criterion_re = nn.MSELoss()
        loss_re1 = criterion_re(rRNA, RNASeq)
        loss_re2 = criterion_re(rmiRNA, miRNA)

        # InfoNCE loss
        loss_info = criterion(output, target)

        real_batch_size = ystatus.shape[0]
        R_matrix_batch_train = torch.tensor(np.zeros([real_batch_size, real_batch_size], dtype=int),
                                            dtype=torch.float).to(device)
        for i in range(real_batch_size):
            R_matrix_batch_train[i,] = torch.tensor(np.array(list(map(int, (ytime >= ytime[i])))))
        theta = model.get_theta(RNASeq, miRNA)
        exp_theta = torch.reshape(torch.exp(theta), [real_batch_size])
        theta = torch.reshape(theta, [real_batch_size])

        loss_cox = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta,
                                                                                R_matrix_batch_train), dim=1))),
                                         torch.reshape(ystatus, [real_batch_size])))


        # ProtoNCE loss
        loss_proto = 0
        for proto_out, proto_target in zip(output_proto, target_proto):
            loss_proto += criterion(proto_out, proto_target)
            accp = accuracy(proto_out, proto_target)[0]
            acc_proto.update(accp[0], RNASeq[0].size(0))

        # average loss across all sets of prototypes
        loss_proto = loss_proto / len(num_cluster)
        
        loss = loss_cox + loss_info + loss_proto + loss_re1 + loss_re2
        # print('loss_proto:', loss_proto.item())
        with open(result_path + str(seed) + CancerName + 'Loss' + '.log', 'a') as f:
            f.writelines(
                'epoch: {} loss:{:.6f}  loss_cox:{:.6f}  loss_info:{:.6f}'
                'loss_proto:{:.6f} loss_re1:{:.6f} loss_re2:{:.6f}  \n'.format(epoch, loss.item(), loss_cox.item(), loss_info.item(),
                                                                               loss_proto.item(),loss_re1.item(), loss_re2.item()
                                                                              )
            )

        if np.all(ystatus.cpu().numpy() == 0):
            with open(result_path + str(seed) + CancerName + 'Train' + '.log', 'a') as f:
                f.writelines('epoch: ' + str(epoch) + ' ystatus_train all 0' + '\n')
            continue
        else:
            cind_train = CIndex(theta, ytime, ystatus)
            AUC_train = AUC(theta, ytime, ystatus)
            # print('batch_idx:', batch_idx, 'cind_train:', cind_train)
            with open(result_path + str(seed) + CancerName + 'Train' + '.log', 'a') as f:
                f.writelines(
                    'epoch: {} batch_idx:{} cind_train:{:.4f} AUC_train: {:.4f}\n'.format(epoch, batch_idx,
                                                                                          cind_train,
                                                                                          AUC_train)
                )

        losses.update(loss.item(), RNASeq[0].size(0))
       
        # Update meta-parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            progress.display(batch_idx)


def eval_model(eval_loader, model):
    # max_cind = 0
    for batch_idx, (RNASeq, miRNA, ystatus, ytime) in enumerate(eval_loader):
        model.eval()
        with torch.no_grad():

            real_batch_size = RNASeq.shape[0]

            theta = model.get_theta(RNASeq, miRNA)
            theta = torch.reshape(theta, [real_batch_size])
            if np.all(ystatus.cpu().numpy() == 0):
                
                return 0, 0
            else:
                cind_val = CIndex(theta, ytime, ystatus)
                auc_val = AUC(theta, ytime, ystatus)
                # print('epoch:', epoch, 'batch_idx:', batch_idx, 'cind_val:', cind_val)
                #with open(result_path + str(seed) + CancerName + 'Val' + '.log', 'a') as f:
                    #f.writelines(
                     #   'epoch: {} cind:{:.4f} AUC: {:.4f}\n'.format(epoch, cind_val, auc_val)
                   # )

                return cind_val, auc_val


if __name__ == '__main__':
    main()
