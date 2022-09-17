import torch
from torchstat import stat
from KF_shuffle_model import shufflenet_v2_x1_0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "weights/model-29.pth"
model = shufflenet_v2_x1_0(num_classes=6)
model.load_state_dict(torch.load(model_path, map_location=device))
# 导入模型，输入一张输入图片的尺寸
stat(model, (3, 224, 224))
