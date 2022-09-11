import torchvision.transforms
from my_dataset import MyDataSetRGB, MyDataSetL
from utils import read_mydata, get_stat3, get_stat1

images_path, images_label = read_mydata("KMU-FED", "KMU-FED")
dataset = MyDataSetRGB(images_path=images_path,
                       images_class=images_label,
                       transform=torchvision.transforms.ToTensor())
mean3, std3 = get_stat3(dataset)
print(mean3)
print(std3)

# images_path, images_label = read_mydata("CK+", "CK+")
# dataset = MyDataSetL(images_path=images_path,
#                     images_class=images_label,
#                     transform=torchvision.transforms.ToTensor())
# mean1, std1 = get_stat1(dataset)
# print(mean1)
# print(std1)
