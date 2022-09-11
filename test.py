import argparse
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from CKPLUS_ShuffleNet_V2_model import shufflenet_v2_x1_0
from my_dataset import MyDataSetL
from utils import read_split_data, train_one_epoch, evaluate, plot_accuracy, read_mydata, data_set_split


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--model_path', type=str, default="./weights/model-9.pth")
    # shufflenetv2_x1.0 官方权重下载地址
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='shufflenetv2_x1-5666bf0f80.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    return args


def main(args):
    test_images_path, test_images_label = read_mydata("CK+_classification/test", "test")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    # shuffle = shufflenet_v2_x1_0().to(device)
    # model = shuffle.load_state_dict(torch.load(args.model_path))
    model = shufflenet_v2_x1_0(num_classes=7).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(model)
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])
    test_dataset = MyDataSetL(images_path=test_images_path,
                              images_class=test_images_label,
                              transform=data_transform)
    batch_size = args.batch_size
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             collate_fn=test_dataset.collate_fn)

    total_accuracy_val = []
    acc_val = evaluate(model=model,
                       data_loader=test_loader,
                       device=device)
    accuracy_val = round(acc_val, 3)
    total_accuracy_val.append(accuracy_val)
    print("[test_accuracy:{}".format(accuracy_val))


if __name__ == '__main__':
    start = time.perf_counter()
    args = parse()
    main(args)
    end = time.perf_counter()
    print('Running time: {} Minutes, {} Seconds'.format((end - start) // 60, (end - start) % 60))
