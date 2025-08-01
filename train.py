import argparse
import time
import torch.nn.functional as F
from model import PCH, EvidenceNet
from utils import *
from data import *
from lossss import MultiSimilarityLoss


def train(args, dset):
    assert dset.I_tr.shape[0] == dset.T_tr.shape[0]
    assert dset.I_tr.shape[0] == dset.L_tr.shape[0]
    logName = args.dataset + '_' + str(args.nbit)
    log = logger(logName)
    log.info('Training Stage...')
    loss_l2 = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    model = PCH(args=args)
    model.train().cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr}])
    start_time = time.time() * 1000
    MSL = MultiSimilarityLoss()

    train_loader = data.DataLoader(my_dataset(dset.I_tr, dset.T_tr, dset.L_tr),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    

    for epoch in range(args.epochs):
        for i, (idx, img_feat, txt_feat, label) in enumerate(train_loader):
            _, aff_norm, aff_label = affinity_tag_multi(label.numpy(), label.numpy())
            
            img_feat = img_feat.cuda()
            txt_feat = txt_feat.cuda()
            label = label.cuda()
            

            aff_label = torch.Tensor(aff_label).cuda()

            optimizer.zero_grad()
            H, pred = model(img_feat, txt_feat, label)
            H_norm = F.normalize(H)
            center = model.centroids.to(dtype=torch.float32).cuda()
            cen_loss = center_loss(center)
            similarity_loss = loss_l2(H_norm.mm(H_norm.t()), aff_label)
            codeloss = self.MSL(H, label)
            loss = cen_loss * args.param_cen + similarity_loss * args.param_sim + codeloss * args.param_it
            loss.backward()
            optimizer.step()
            if (i + 1) == len(train_loader) and (epoch + 1) % 2 == 0:
                log.info('Epoch [%3d/%3d], Loss: %.4f, Loss-C: %.4f, Loss-S: %.4f, Loss-IT: %.4f'
                          % (epoch + 1, args.epochs, loss.item(),
                             cen_loss.item() * args.param_cen,
                             similarity_loss.item() * args.param_sim,
                             code_cen_loss.item() * args.param_it))  

    end_time = time.time() * 1000
    elapsed = (end_time - start_time) / 1000
    log.info('Training Time: %.4f' % (elapsed))
    return model



if __name__ == '__main__':
    
    model = train(args, dset)
    eval(model, dset, args)
