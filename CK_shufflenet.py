import os
import math
import argparse
import shutil
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from utils import train_one_epoch, evaluate, plot_accuracy, read_to_csv
# from ShuffleNetV2_model import shufflenet_v2_x1_0
from CKPLUS_ShuffleNet_V2_model import shufflenet_v2_x1_0

train_loader, val_loader = read_to_csv("CK+", "CK+.csv")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    logs_path = "runs"
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    tb_writer = SummaryWriter(logs_path)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    # 如果存在预训练权重则载入
    model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    total_accuracy_val = []
    total_accuracy_train = []
    for epoch in range(args.epochs):
        # train
        mean_loss, acc_train = train_one_epoch(model=model,
                                               optimizer=optimizer,
                                               data_loader=train_loader,
                                               device=device,
                                               epoch=epoch)
        accuracy_train = round(acc_train, 3)
        total_accuracy_train.append(accuracy_train)

        scheduler.step()

        # validate
        acc_val = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        accuracy_val = round(acc_val, 3)
        total_accuracy_val.append(accuracy_val)
        print("[epoch {}] train_accuracy: {} test_accuracy:{}".format(epoch, accuracy_train, accuracy_val))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc_val, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    plot_accuracy(args.epochs, total_accuracy_train, total_accuracy_val, "ShuffleNet_V2_CK+")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="CK+")

    # shufflenetv2_x1.0 官方权重下载地址
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='shufflenetv2_x1-5666bf0f80.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
