import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample

class PCLSurv(nn.Module):

    def __init__(self, RNAseq_dim, miRNA_dim):
        nn.Module.__init__(self)

        self.RNASeq_encoding = nn.Sequential(nn.Linear(RNAseq_dim, 200),
                                             nn.ReLU(),
                                             nn.Linear(200, 50),
                                             nn.ReLU(),
                                             nn.Linear(50, 50)
                                             )
        self.RNASeq_decoding = nn.Sequential(nn.Linear(50, 50),
                                             nn.ReLU(),
                                             nn.Linear(50, 200),
                                             nn.ReLU(),
                                             nn.Linear(200, RNAseq_dim),
                                             nn.Sigmoid()
                                             )
        self.miRNA_encoding = nn.Sequential(nn.Linear(miRNA_dim, 200),
                                            nn.ReLU(),
                                            nn.Linear(200, 50),
                                            nn.ReLU(),
                                            nn.Linear(50, 50)
                                            )
        self.miRNA_decoding = nn.Sequential(nn.Linear(50, 50),
                                            nn.ReLU(),
                                            nn.Linear(50, 200),
                                            nn.ReLU(),
                                            nn.Linear(200, miRNA_dim),
                                            nn.Sigmoid()
                                            )

        self.r = 64
        self.m = 0.9
        self.T = 0.2
        self.dim = 50

        self.con_linear = nn.Linear(100, 50)
        self.RNASeq_linear = nn.Linear(50, 50)
        self.miRNA_linear = nn.Linear(50, 50)
        self.bn = nn.BatchNorm1d(60)
        self.dropout = nn.Dropout(0.3)
        self.feature_fusion = nn.Linear(60, 30)
        self.Cox = nn.Sequential(nn.Linear(30, 1))


    def forward(self, omics_q, omics_k, is_eval=False, cluster_result=None, gpu_id=None, pre=None, index=None):
        """
        Input:
            omics_q: a batch of query omics data RNA
            omics_k: a batch of key omics data   miRNA
            is_eval: return embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        self.omics_q = self.RNASeq_encoding(omics_q)
        q = self.omics_q
        RNA_new = self.RNASeq_decoding(q)
        self.omics_k = self.miRNA_encoding(omics_k)
        k = self.omics_k
        miRNA_new = self.miRNA_decoding(k)


        fbn_feature = self.get_fbn_feature(q, k)  # 64*10
        X = torch.cat((F.normalize(q + k), fbn_feature), 1)

        if is_eval:
            return X

        # compute logits
        # Einstein sum is more intuitive

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: Nxr
        l_neg_A = torch.einsum('nc,mc->nm', [q, q])  # (N, N)
        l_neg_B = torch.einsum('nc,mc->nm', [q, k])  # (N, N)

        mask = torch.eye(q.shape[0], dtype=torch.bool).cuda()
        l_neg_A = l_neg_A[~mask].reshape(q.shape[0], -1)  # (N, N-1)
        l_neg_B = l_neg_B[~mask].reshape(q.shape[0], -1)  # (N, N-1)

        l_neg = torch.cat([l_neg_A, l_neg_B], dim=1)  # (N, 2N-2)

        logits = torch.cat([l_pos, l_neg], dim=1)  # (N, 1 + 2N-2)

        # apply temperature
        logits /= self.T

        # labels: positive indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(gpu_id)


        # prototypical contrast

        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            for n, (can2cluster, prototypes, density) in enumerate(
                    zip(cluster_result['can2cluster'], cluster_result['centroids'], cluster_result['density'])):
                pos_prototypes = prototypes[can2cluster] 
                l_pro_pos = torch.einsum('nc,nc ->n', X, pos_prototypes) 
                l_pro_pos = l_pro_pos.unsqueeze(1)
                idx_number = torch.unique(can2cluster) 
                neg_id = torch.stack([idx_number[idx_number != val] for val in can2cluster]) 
                neg_prototypes = prototypes[neg_id] 
                l_pro_neg = torch.einsum('bik,bk->bi', neg_prototypes, X) 
                logits_proto = torch.cat([l_pro_pos, l_pro_neg], dim=1)
                labels_proto = torch.zeros(logits.shape[0], dtype=torch.long).cuda(gpu_id)

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

            return RNA_new, miRNA_new, logits, labels, proto_logits, proto_labels



    def get_fbn_feature(self, RNASeq_feature, miRNA_feature):
        temp_fea = torch.sum(torch.reshape(self.RNASeq_linear(self.dropout(RNASeq_feature)), (-1, 5, 10)) *
                             torch.reshape(self.miRNA_linear(miRNA_feature), (-1, 5, 10)), dim=1)
                            
        return torch.sqrt(F.relu(temp_fea)) - torch.sqrt(F.relu(-temp_fea))

    def get_theta(self, RNA, miRNA):

        RNA = self.RNASeq_encoding(RNA)
        miRNA = self.miRNA_encoding(miRNA)
        fbn_feature = self.get_fbn_feature(RNA, miRNA)
        X = torch.cat((F.normalize(RNA + miRNA), fbn_feature), 1)
        theta = self.Cox(self.feature_fusion(self.bn(X)))

        return theta


