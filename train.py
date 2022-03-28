# -*- coding: utf-8 -*-
import os
import torch
import transforms
from intu_dataset import DataSet
from src import SSD300, Backbone
import train_utils.train_eval_utils as utils
from train_utils import get_coco_api_from_dataset
import datetime
import time

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def create_model(num_classes=1, num_names=4, is_intu=False, device=torch.device('cpu')):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes, num_names=num_names, is_intu=is_intu)
    pre_ssd_path = "./src/nvidia_ssdpyt_fp32.pt"
    if os.path.exists(pre_ssd_path) is False:
        raise FileNotFoundError("nvidia_ssdpyt_fp32.pt not find in {}".format(pre_ssd_path))
    pre_model_dict = torch.load(pre_ssd_path, map_location=device)
    pre_weights_dict = pre_model_dict["model"]
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
    return model.to(device)


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.SSDCropping(),
                                     transforms.Resize(),
                                     transforms.ColorJitter(),
                                     transforms.ToTensor(),
                                     transforms.Normalization(),
                                     transforms.AssignGTtoDefaultBox()]),
        "val": transforms.Compose([transforms.Resize(),
                                   transforms.ToTensor(),
                                   transforms.Normalization()])
    }

    VOC_root = parser_data.data_path
    if os.path.exists(os.path.join(VOC_root, "intudataset_classification")) is False:
        raise FileNotFoundError("intudataset_classification dose not in path:'{}'.".format(VOC_root))

    train_dataset = DataSet(VOC_root, data_transform['train'], train_set='train.txt')
    batch_size = parser_data.batch_size
    assert batch_size > 1, "batch size must be greater than 1"
    drop_last = True if len(train_dataset) % batch_size == 1 else False
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn,
                                                    drop_last=drop_last)

    val_dataset = DataSet(VOC_root, data_transform['val'], train_set='val.txt')
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes + 1,
                         num_names=args.num_names + 1,
                         is_intu=args.is_intu,
                         device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)

    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    Accuracy_list = []

    val_data = get_coco_api_from_dataset(val_data_loader.dataset)
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        mean_loss, lr = utils.train_one_epoch(model=model, optimizer=optimizer,
                                              data_loader=train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        lr_scheduler.step()
        coco_info, acc_name, y_pred, y_true, coco_ious, scores = utils.evaluate(model=model, data_loader=val_data_loader,
                                                             device=device, data_set=val_data)

        Accuracy_list.append(acc_name)

        with open(results_file, "a") as f:
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item(), lr]]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
            f.write('mAP' + '' + str(coco_info[1]) + "\n")
            f.write('acc_name' + ' ' + str(acc_name) + "\n")
        val_map.append(coco_info[1])

    save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': parser_data.epochs}
    torch.save(save_files, "./save_weights/ssd.pth")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--num_classes', default=1, type=int, help='num_classes')
    parser.add_argument('--num_names', default=4, type=int, help='num_names')
    parser.add_argument('--data-path', default='./', help='dataset')
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=83, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--is_intu', default=True, type=bool, help='set is_intu')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
