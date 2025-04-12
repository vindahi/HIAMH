import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F
import math
import numpy as np



class Fusion(nn.Module):
    def __init__(self, fusion_dim, nbit):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        self.hash = nn.Sequential(
            nn.Linear(fusion_dim, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh()
        )

    def forward(self, x, y):
        fused_feat = self.fusion(torch.cat([x, y], dim=-1))
        hash_code = self.hash(fused_feat)
        return hash_code


class PCH(nn.Module):
    def __init__(self, args):
        super(PCH, self).__init__()
        self.image_dim = args.image_dim
        self.text_dim = args.text_dim

        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        self.nbit = int(args.nbit)
        self.classes = args.classes
        self.batch_size = args.batch_size
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]

        self.fusionnn = Fusion(fusion_dim=self.common_dim, nbit=self.nbit)

        self.imageMLP = MLP(hidden_dim=self.img_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)

        self.textMLP = MLP(hidden_dim=self.txt_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)
        self.imglabelatt = AttentionLayer(self.common_dim, self.classes, self.common_dim)
        self.txtlabelatt = AttentionLayer(self.common_dim, self.classes, self.common_dim)

        self.Fcfusion = nn.Linear(2 * self.common_dim, self.common_dim)


        self.fusion_layer = nn.Sequential(
            nn.Linear(self.nbit, self.common_dim),
            nn.ReLU()
        )
        self.hash_output = nn.Sequential(
            nn.Linear(self.common_dim, self.nbit),
            nn.Tanh())
        self.hashfc = nn.Sequential(
            nn.Linear(self.nbit, 2 * self.nbit),
            nn.Sigmoid(),
            nn.Linear(2 * self.nbit, self.nbit))
        
        self.weight = nn.Parameter(torch.randn(self.nbit))  # 可学习的权重
        nn.init.normal_(self.weight, 0.25, 1/self.nbit)
        self.centroids = nn.Parameter(torch.randn(self.classes, self.nbit)).to(dtype=torch.float32)
        self.classify = nn.Linear(self.nbit, self.classes)

    def forward(self, image, text, label):
        self.batch_size = len(image)
        imageH = self.imageMLP(image)
        textH = self.textMLP(text)
        imagefine = self.imglabelatt(imageH, label)
        textfine = self.txtlabelatt(textH, label)

        img_feature = self.Fcfusion(torch.cat((imageH, imagefine), 1))
        text_feature = self.Fcfusion(torch.cat((textH, textfine), 1))

        fused_fine = self.fusionnn(img_feature, text_feature)

        cfeat_concat = self.fusion_layer(fused_fine)    
        code = self.hash_output(cfeat_concat)
        return code, self.classify(code)


class AttentionLayer(nn.Module):
    def __init__(self, data_dim, label_dim, hidden_dim, n_heads=4):
        super(AttentionLayer, self).__init__()

        assert hidden_dim % n_heads == 0

        self.data_dim = data_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(label_dim, hidden_dim)
        self.fc_k = nn.Linear(data_dim, hidden_dim)
        self.fc_v = nn.Linear(data_dim, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        self.dense = nn.Linear(hidden_dim, data_dim)    
        self.bn = nn.BatchNorm1d(data_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data_tensor, label_tensor):
        batch_size = data_tensor.shape[0]

        Q = self.fc_q(label_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        K = self.fc_k(data_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        V = self.fc_v(data_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()

        att_map = torch.softmax((torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale), dim=-1)
        output = torch.matmul(att_map, V).view(batch_size, -1)

        output = self.dense(output)
        output = self.bn(output)
        output = self.relu(output)

        return output






class EvidenceNet(nn.Module):
    def __init__(self, bit_dim, tau_coff):
        """
        :param bit_dim: bit number of the final binary code
        """
        super(EvidenceNet, self).__init__()
        self.tau = tau_coff

        self.acquisition = nn.Sequential(
            nn.Linear(bit_dim, bit_dim)  # Single acquisition layer
        )    

        self.getPosE = nn.Sequential(
            nn.Linear(bit_dim, 2 * bit_dim),
            nn.ReLU(),
            nn.Linear(2 * bit_dim, 1)
        )

    def forward(self, hash_code):
        STE = lambda x: (x.sign() / np.sqrt(hash_code.shape[1]) - x).detach() + x
        
        hash_STE = STE(hash_code)

        hash_composed = self.get_abc(hash_STE)
        negE = self.getPosE(hash_composed)
        posE = hash_STE @ hash_STE.T  # Self similarity


        posE = torch.exp(torch.clamp((posE) / self.tau, -15, 15).view(-1, 1))
        negE = torch.exp(torch.clamp((negE) / self.tau, -15, 15).view(-1, 1))
        return torch.cat([posE, negE], dim=1)

    def get_abc(self, hash_STE):
        ni = hash_STE.size(0)
        di = hash_STE.size(1)
        
        hash_STE = hash_STE.unsqueeze(1).expand(ni, ni, di)
        hash_STE = hash_STE.reshape(-1, di)
        
        return hash_STE



# class PCH(nn.Module):
#     def __init__(self, args):
#         super(PCH, self).__init__()
#         self.image_dim = args.image_dim
#         self.text_dim = args.text_dim

#         self.img_hidden_dim = args.img_hidden_dim
#         self.txt_hidden_dim = args.txt_hidden_dim
#         self.common_dim = args.img_hidden_dim[-1]
#         self.nbit = int(args.nbit)
#         self.classes = args.classes
#         self.batch_size = args.batch_size
        
#         assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]

#         # self.dropout = args.dropout
#         self.fusionnn = Fusion(fusion_dim=self.common_dim, nbit=self.nbit)

#         self.imageMLP = MLP(hidden_dim=self.img_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)

#         self.textMLP = MLP(hidden_dim=self.txt_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)

#         self.ifeat_gate = nn.Sequential(
#             nn.Linear(self.common_dim, self.common_dim*2),
#             nn.ReLU(),
#             nn.Linear(self.common_dim*2, self.common_dim), 
#             nn.Sigmoid())
#         self.tfeat_gate = nn.Sequential(
#             nn.Linear(self.common_dim, self.common_dim*2),
#             nn.ReLU(),
#             nn.Linear(self.common_dim*2, self.common_dim),
#             nn.Sigmoid())
#         self.pro = nn.Sequential(
#             nn.Linear(self.common_dim, self.common_dim*2),
#             nn.ReLU(),
#             nn.Linear(self.common_dim*2, self.nbit), 
#             nn.BatchNorm1d(self.nbit),
#             nn.Sigmoid())
#         self.activation = nn.ReLU()
#         self.neck = nn.Sequential(
#             nn.Linear(self.common_dim,self.common_dim*4),
#             nn.ReLU(),
#             nn.Dropout(args.dropout),
#             nn.Linear(self.common_dim*4,self.common_dim)
#         )

#         self.fusion_layer = nn.Sequential(
#             nn.Linear(self.nbit, self.common_dim),
#             nn.ReLU()
#         )

#         self.hash_output = nn.Sequential(
#             nn.Linear(self.common_dim, self.nbit),
#             nn.Tanh())
#         self.classify = nn.Linear(self.nbit, self.classes)

#     def forward(self, image, text, tgt=None):
#         self.batch_size = len(image)
#         imageH = self.imageMLP(image)#nbit length
#         textH = self.textMLP(text)
#         ifeat_info = self.ifeat_gate(imageH)#nbit length
#         tfeat_info = self.tfeat_gate(textH)
#         image_feat = ifeat_info*imageH
#         text_feat = tfeat_info*textH
        
#         # pimage = self.pro(image_feat) + torch.randn_like(self.pro(image_feat)) * 1
#         # ptext = self.pro(text_feat) + torch.randn_like(self.pro(text_feat)) * 1

#         pimage = self.pro(image_feat) 
#         ptext = self.pro(text_feat) 
#         fused_fine = self.fusionnn(image_feat, text_feat)

#         cfeat_concat = self.fusion_layer(fused_fine)
#         cfeat_concat = self.activation(cfeat_concat)
#         nec_vec = self.neck(cfeat_concat)     
#         code = self.hash_output(nec_vec)   
#         return pimage, ptext, code, self.classify(code)




class classone(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter((torch.randn(args.classes, args.nbit) / 8))
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mrg = 1.0

    def forward(self, feature_1, predict_1, label_1):
        feature_all = feature_1
        label_all = label_1
        proxies = F.normalize(self.proxies, p=2, dim=-1)
        feature_all = F.normalize(feature_all, p=2, dim=-1)

        D_ = torch.cdist(feature_all, proxies) ** 2

        mrg = torch.zeros_like(D_)
        mrg[label_all == 1] = mrg[label_all == 1] + self.mrg
        D_ = D_ + mrg

        p_loss = torch.sum(-label_all * F.log_softmax(-D_, 1), -1).mean()

        d_loss = self.cross_entropy(predict_1, torch.argmax(label_1, -1))

        loss = p_loss + d_loss
        return loss


