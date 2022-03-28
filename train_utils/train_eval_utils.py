import math
import sys
import time
import torch
from train_utils import get_coco_api_from_dataset, CocoEvaluator
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, warmup=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 5.0 / 10000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.stack(images, dim=0)
        boxes = []
        labels = []
        name = []
        img_id = []
        for t in targets:
            boxes.append(t['boxes'])
            labels.append(t['labels'])
            name.append(t['name'])
            img_id.append(t["image_id"])
        targets = {"boxes": torch.stack(boxes, dim=0),
                   "labels": torch.stack(labels, dim=0),
                   "name": torch.stack(name, dim=0),
                   "image_id": torch.as_tensor(img_id)}

        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        losses_dict = model(images, targets)
        losses = losses_dict["total_losses"]
        # reduce losses over all GPUs for logging purpose
        losses_dict_reduced = utils.reduce_dict(losses_dict)
        losses_reduce = losses_dict_reduced["total_losses"]
        loss_value = losses_reduce.detach()
        mloss = (mloss * i + loss_value) / (i + 1)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(**losses_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr



@torch.no_grad()
def evaluate(model, data_loader, device, data_set=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    if data_set is None:
        data_set = get_coco_api_from_dataset(data_loader.dataset)

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(data_set, iou_types)
    acc_name = 0.0
    y_true = []
    y_pred = []
    scores_max = []
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = torch.stack(images, dim=0).to(device)
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        model_time = time.time()
        results, names = model(images, targets=None)
        model_time = time.time() - model_time
        outputs = []
        for index, (bboxes_out, labels_out, scores_out) in enumerate(results):
            scores_out1 = scores_out.cpu().numpy()
            if len(scores_out1) == 0:
                continue
            else:
                scores_max.append(scores_out1[0])
            height_width = targets[index]["height_width"]
            bboxes_out[:, [0, 2]] = bboxes_out[:, [0, 2]] * height_width[1]
            bboxes_out[:, [1, 3]] = bboxes_out[:, [1, 3]] * height_width[0]

            info = {"boxes": bboxes_out.to(cpu_device),
                    "labels": labels_out.to(cpu_device),
                    "scores": scores_out.to(cpu_device)}
            outputs.append(info)
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        predict_name = torch.max(names, dim=1)[1]
        target_name = []
        for target in targets:
            target_name.append(target['name'].item())
            y_true.append(target['name'].item())
        target_name1 = torch.tensor(target_name)
        acc_name += torch.eq(predict_name, target_name1.to(device)).sum().item()/len(results)
        list1 = predict_name.cpu().numpy().tolist()
        for a in list1:
            y_pred.append(a)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    coco_ious = coco_evaluator.ious
    acc_name = acc_name/len(data_loader)
    return coco_info, acc_name, y_pred, y_true, coco_ious, scores_max


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
