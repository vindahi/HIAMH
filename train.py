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
    # log.info('mlpdrop: %f', (args.mlpdrop))
    # log.info('drop: %f', (args.dropout))
    # log.info('epochs: %f', (args.epochs))
    # log.info('param_sim: %f', (args.param_sim))
    # log.info('param_it: %f', (args.param_it))
    # log.info('param_sup: %f', (args.param_sup))
    loss_l2 = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    model = PCH(args=args)
    # evidence_model = EvidenceNet(args.nbit,args.tau).cuda()
    model.train().cuda()
    # evidence_model.train().cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr}])
    # optimizer_evi = torch.optim.Adam([{'params': evidence_model.parameters(), 'lr': args.lr}])

    start_time = time.time() * 1000

    MSL = MultiSimilarityLoss()
    # criterion = SupConLoss(loss=args.loss, temperature=args.temp,  data_class=args.classes).cuda()



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

            code_center = H.mm(center.t())
            constr_loss = bce_loss(code_center, label)

            clf_loss = loss_l2(torch.sigmoid(pred), label)

            similarity_loss = loss_l2(H_norm.mm(H_norm.t()), aff_label)

            code_cen_loss = code_center_loss(H, center, label)



            # loss = clf_loss * args.param_clf + similarity_loss * args.param_sim + it_loss * args.param_it
            # loss = clf_loss * args.param_clf + similarity_loss * args.param_sim + losssup * args.param_sup + lossiitt * args.param_it
            loss = clf_loss * args.param_clf + similarity_loss * args.param_sim + cen_loss * args.param_cen + code_cen_loss * args.param_it + constr_loss * args.param_sup
            ####loss = clf_loss * args.param_clf + similarity_loss * args.param_sim + cen_loss * args.param_cen + code_cen_loss * args.param_it
            #loss = clf_loss * args.param_clf + similarity_loss * args.param_sim + cen_loss * args.param_cen + constr_loss * args.param_sup
            #loss = clf_loss * args.param_clf  + cen_loss * args.param_cen + code_cen_loss * args.param_it + constr_loss * args.param_sup
            loss.backward()
            optimizer.step()
            if (i + 1) == len(train_loader) and (epoch + 1) % 2 == 0:
                log.info('Epoch [%3d/%3d], Loss: %.4f, Loss-C: %.4f, Loss-S: %.4f, Loss-CEN: %.4f, Loss-IT: %.4f, Loss-SUP: %.4f'
                          % (epoch + 1, args.epochs, loss.item(),
                             clf_loss.item() * args.param_clf,
                             similarity_loss.item() * args.param_sim,
                             cen_loss.item() * args.param_cen,
                             code_cen_loss.item() * args.param_it,
                             constr_loss.item() * args.param_sup))  

    end_time = time.time() * 1000
    elapsed = (end_time - start_time) / 1000
    log.info('Training Time: %.4f' % (elapsed))


    return model






def eval(model, dset, args):
    model.eval()
    rank_index = get_rank(model)
    logName = args.dataset + '_' + str(args.nbit)
    log = logger(logName)
    assert dset.I_db.shape[0] == dset.T_db.shape[0]
    assert dset.I_db.shape[0] == dset.L_db.shape[0]

    retrieval_loader = data.DataLoader(my_dataset(dset.I_db, dset.T_db, dset.L_db),
                                       batch_size=args.eval_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True)

    retrievalP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, label) in enumerate(retrieval_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        label = label.cuda()
        H, _ = model(img_feat, txt_feat, label)
        retrievalP.append(H.data.cpu().numpy())

    retrievalH = np.concatenate(retrievalP)
    retrievalCode = np.sign(retrievalH)

    end_time = time.time() * 1000
    retrieval_time = end_time - start_time

    log.info('Query size: %d' % (dset.I_te.shape[0]))
    assert dset.I_te.shape[0] == dset.T_te.shape[0]
    assert dset.I_te.shape[0] == dset.L_te.shape[0]

    val_loader = data.DataLoader(my_dataset(dset.I_te, dset.T_te, dset.L_te),
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

    valP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, label) in enumerate(val_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        label = label.cuda()
        H, _ = model(img_feat, txt_feat, label)
        valP.append(H.data.cpu().numpy())

    valH = np.concatenate(valP)
    valCode = np.sign(valH)



    end_time = time.time() * 1000
    query_time = end_time - start_time
    log.info('[Retrieval time] %.4f, [Query time] %.4f' % (retrieval_time / 1000, query_time / 1000))
    if args.save_flag:
        map = calculate_map(qu_B=valCode.astype(np.int8), re_B=retrievalCode.astype(np.int8), qu_L=dset.L_te, re_L=dset.L_db)
        log.info('[MAP] %.4f' % (map))
        if isinstance(valCode, torch.Tensor):
            valCode_np = valCode.cpu().numpy().astype(np.int8)
        else:
            valCode_np = valCode.astype(np.int8)

        if isinstance(retrievalCode, torch.Tensor):
            retrievalCode_np = retrievalCode.cpu().numpy().astype(np.int8)
        else:
            retrievalCode_np = retrievalCode.astype(np.int8)
        
        if isinstance(rank_index, torch.Tensor):
            rank_index_np = rank_index.cpu().numpy()
        else:
            rank_index_np = rank_index
        # for bit in [128, 64, 32, 16]:
        #     bit_ind = rank_index_np[:bit]
        #     valCodeind = valCode_np[:, bit_ind]
        #     retrievalCodeind = retrievalCode_np[:, bit_ind]
        #     map = calculate_map(qu_B=valCodeind.astype(np.int8), re_B=retrievalCodeind.astype(np.int8), qu_L=dset.L_te, re_L=dset.L_db)
        #     log.info('[IndMAP] %d-bit %.4f' % (bit, map))
        for start_bit in range(1, 120, 10):
            end_bit = min(start_bit + 10, args.nbit)  # 确保不超过总位数
            bit_ind = rank_index_np[start_bit:end_bit]
            valCodeind = valCode_np[:, bit_ind]
            retrievalCodeind = retrievalCode_np[:, bit_ind]
            map = calculate_map(qu_B=valCodeind.astype(np.int8), re_B=retrievalCodeind.astype(np.int8), qu_L=dset.L_te, re_L=dset.L_db)
            log.info('[IndMAP] %d-%d bit %.4f' % (start_bit, end_bit, map)) # start_bit + 1 使得输出从1开始，更符合习惯


    return 0





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    ## Net basic params
    parser.add_argument('--model', type=str, default='FSFH', help='Use GMMH.')
    parser.add_argument('--self_paced',  type=bool, default='True', help='--self_paced learning schedule')
    parser.add_argument('--epochs', type=int, default=140, help='Number of student epochs to train.')
    parser.add_argument('--epochs_pre', type=int, default=100, help='Epoch to learn the hashcode.')
    parser.add_argument('--nbit', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.8, help='')
    parser.add_argument('--mlpdrop', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--loss', type=str, default='p2p', help='different kinds of loss')
    parser.add_argument('--temp', type=float, default=0.3, help='temperature')
    parser.add_argument("--tau",type=float,default=0.2)

    parser.add_argument('--nhead', type=int, default=1, help='"nhead" in Transformer.')
    parser.add_argument('--num_layer', type=int, default=2, help='"num_layer" in Transformer.')
    parser.add_argument('--trans_act', type=str, default='gelu', help='"activation" in Transformer.')

    
    ## Data params
    parser.add_argument('--dataset', type=str, default='flickr', help='coco/nuswide/flickr')
    parser.add_argument('--classes', type=int, default=24)
    parser.add_argument('--image_dim', type=int, default=4096)
    parser.add_argument('--text_dim', type=int, default=1386)

    ## Net latent dimension params
    # COCO: 128 Flickr: 256
    parser.add_argument('--img_hidden_dim', type=list, default=[2048, 128], help='Construct imageMLP')
    parser.add_argument('--txt_hidden_dim', type=list, default=[1024, 128], help='Construct textMLP')
    

    ## Loss params
    parser.add_argument('--param_dn', type=float, default=0.000001)
    parser.add_argument('--param_qmi', type=float, default=0.000001)
    parser.add_argument('--param_clf', type=float, default=1)
    parser.add_argument('--param_sim', type=float, default=1)
    parser.add_argument('--param_cluster', type=float, default=0.01)
    parser.add_argument('--param_it', type=float, default=0.01)
    parser.add_argument('--param_sup', type=float, default=0.0001)
    parser.add_argument('--param_cen', type=float, default=0.01)

    ## Flag params
    parser.add_argument('--save_flag', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    seed_setting(args.seed)

    dset = load_data(args.dataset)
    print('Train size: %d, Retrieval size: %d, Query size: %d' % (dset.I_tr.shape[0], dset.I_db.shape[0], dset.I_te.shape[0]))
    print('Image dimension: %d, Text dimension: %d, Label dimension: %d' % (dset.I_tr.shape[1], dset.T_tr.shape[1], dset.L_tr.shape[1]))

    args.image_dim = dset.I_tr.shape[1]
    args.text_dim = dset.T_tr.shape[1]
    args.classes = dset.L_tr.shape[1]

    args.img_hidden_dim.insert(0, args.image_dim)
    args.txt_hidden_dim.insert(0, args.text_dim)


    model = train(args, dset)
    eval(model, dset, args)