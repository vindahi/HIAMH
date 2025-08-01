import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels):
        # Ensure input dimensions match
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        
        batch_size = feats.size(0)

        # Normalize features and compute similarity matrix
        feats = F.normalize(feats, p=2, dim=1)
        sim_mat = torch.matmul(feats, feats.t())

        # Create label indicator matrix
        labels = labels @ labels.t() > 0

        epsilon = 1e-5
        loss_list = []

        for i in range(batch_size):
            # Extract positive and negative pairs
            pos_pairs = sim_mat[i][labels[i]]
            pos_pairs = pos_pairs[pos_pairs < 1 - epsilon]
            neg_pairs = sim_mat[i][~labels[i]]

            # Skip if no valid pairs
            if torch.numel(pos_pairs) == 0 or torch.numel(neg_pairs) == 0:
                continue

            # Filter pairs based on margin
            neg_pairs = neg_pairs[neg_pairs + self.margin > pos_pairs.min()]
            pos_pairs = pos_pairs[pos_pairs - self.margin < neg_pairs.max()]

            if len(neg_pairs) < 1 or len(pos_pairs) < 1:
                continue

            # Compute positive and negative loss
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pairs - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pairs - self.thresh))))

            loss_list.append(pos_loss + neg_loss)

        # Return zero loss if no valid pairs are found
        if len(loss_list) == 0:
            return torch.zeros([], requires_grad=True)

        # Average loss over batch size
        return sum(loss_list) / batch_size


class SupConLoss(nn.Module):
    def __init__(self, loss='p2p', temperature=0.3, data_class=10):
        super(SupConLoss, self).__init__()
        self.loss = loss
        self.temperature = temperature
        self.data_class = data_class

    def forward(self, features, prototypes, labels=None, epoch=0, args=None):
        # Data-to-data computation
        anchor_feature = features
        contrast_feature = features
        mask = (torch.mm(labels.float(), labels.float().T) > 0).float()
        pos_mask = mask
        neg_mask = 1 - mask

        # Compute similarity matrix for data-to-data
        anchor_dot_contrast = torch.matmul(F.normalize(anchor_feature, dim=1), 
                                           F.normalize(contrast_feature, dim=1).T)
        all_exp = torch.exp(anchor_dot_contrast / self.temperature)
        pos_exp = pos_mask * all_exp
        neg_exp = neg_mask * all_exp

        # Data-to-class computation
        pos_mask2 = labels
        neg_mask2 = 1 - labels
        anchor_dot_prototypes = torch.matmul(F.normalize(anchor_feature, dim=1), 
                                             F.normalize(prototypes, dim=1).T)
        all_exp2 = torch.exp(anchor_dot_prototypes / self.temperature)
        pos_exp2 = pos_mask2 * all_exp2
        neg_exp2 = neg_mask2 * all_exp2

        # Apply self-paced learning adjustments
        if args and args.self_paced:
            if epoch <= int(args.epochs / 3):
                delta = epoch / int(args.epochs / 3)
            else:
                delta = 1
            pos_exp *= torch.exp(-1 - anchor_dot_contrast).detach() ** (delta / 4)
            neg_exp *= torch.exp(-1 + anchor_dot_contrast).detach() ** delta
            pos_exp2 *= torch.exp(-1 - anchor_dot_prototypes).detach() ** (delta / 4)
            neg_exp2 *= torch.exp(-1 + anchor_dot_prototypes).detach() ** delta

        # Compute loss based on mode
        if self.loss == 'p2p':
            loss = -torch.log(pos_exp.sum(1) / (neg_exp.sum(1) + pos_exp.sum(1)))
            return loss.mean()
        elif self.loss == 'p2c':
            loss = -torch.log(pos_exp2.sum(1) / (neg_exp2.sum(1) + pos_exp2.sum(1)))
            return loss.mean()
        else:
            raise ValueError(f"Invalid loss type: {self.loss}")
