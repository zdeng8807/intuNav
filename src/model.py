import torch
from torch import nn, Tensor
from torch.jit.annotations import List
from src.resnet import resnet50
from src.utils import dboxes300_coco, Encoder, PostProcess

class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])
        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21, num_names=6, is_intu=False,):
        super(SSD300, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone not has attribute: out_channel")
        self.feature_extractor = backbone
        self.num_classes = num_classes
        self.num_names = num_names
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractors = []
        confidence_extractors = []
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))
        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.name_fc = nn.Linear(1024, num_names)
        self._init_weights()
        default_box = dboxes300_coco(is_intu)
        self.compute_loss = Loss(default_box)
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)

    def _build_additional_features(self, input_size):
        additional_blocks = []
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            locs.append(l(f).view(f.size(0), 4, -1))
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):
        x = self.feature_extractor(image)
        x_name = self.avgpool(x)
        x_name = torch.flatten(x_name, 1)
        names = self.name_fc(x_name)
        detection_features = torch.jit.annotate(List[Tensor], [])
        detection_features.append(x)

        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            labels_out = targets['labels']
            name_out = targets['name']
            loss = self.compute_loss(locs, confs, names, bboxes_out, labels_out, name_out)
            return {"total_losses": loss}
        results = self.postprocess(locs, confs)
        return results, names


    
class Loss(nn.Module):
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh
        self.location_loss = nn.SmoothL1Loss(reduction='none')
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        self.name_loss = nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self, loc):
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, pname, gloc, glabel, gname):
        mask = torch.gt(glabel, 0)
        pos_num = mask.sum(dim=1)
        vec_gd = self._location_vec(gloc)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)
        loc_loss = (mask.float() * loc_loss).sum(dim=1)
        con = self.confidence_loss(plabel, glabel)
        con_neg = con.clone()
        con_neg[mask] = 0.0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num)
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)
        total_loss = loc_loss + con_loss
        num_mask = torch.gt(pos_num, 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        total_loss = (total_loss * num_mask / pos_num).mean(dim=0)
        name_loss = self.name_loss(pname, gname)
        name_loss = name_loss.mean(dim=0)
        ret = total_loss + name_loss
        return ret

