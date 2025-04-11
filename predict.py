import os
import time

from osgeo import gdal
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet, AttU_Net, DinkNet34,DinkNet34_dsc,MTUnet,MTUNet_dsc,MTUNet_dsc_lite,MTRDnet,MTUnet_Cattention,MTUnet,deeplabv3_resnet50\
,ResUNetFormer,MTB_dsc_1bcam_uom,MTB_dsc_2bcam_uom,MTB_dsc_3bcam_uom,MTB_dsc_4bcam_uom,MTB_dsc_5bcam_uom
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    classes = 1  # exclude background
    modelname = 'unet_fusu'
    # model = MTB_dsc_5bcam_uom(in_channels=6, num_classes=2, base_c=32)

    # model = UNet(in_channels=6, num_classes=2, base_c=16)
    # model = DinkNet34(num_classes=2, num_channels=6)
    model = UNet(in_channels=6, num_classes=2, base_c=16)
    weights_path = f"save_weights/best_unet_fusu.pth"
    img_dir = "E:/02agriculture-road-extraction/test/FUSU/"
    out_path = f"E:/02agriculture-road-extraction/test/FUSU/{modelname}/"
    
    imgs = [i for i in os.listdir(f"{img_dir}") if i.endswith(".tif")]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for img_path1 in imgs:
        img_path = os.path.join(f"{img_dir}", img_path1)
        # roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
        # assert os.path.exists(weights_path), f"weights {weights_path} not found."
        # assert os.path.exists(img_path), f"image {img_path} not found."
        # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

        mean = (0.409, 0.481, 0.424, 0.409, 0.481, 0.424)
        std = (0.22, 0.22, 0.22, 0.22, 0.22, 0.22)
        # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # create model
        # model = MTUNet_dsc(in_channels=6, num_classes=2, base_c=32)
        # load weights
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.to(device)

        # load roi mask
        # roi_img = Image.open(roi_mask_path).convert('L')
        # roi_img = np.array(roi_img)

        # load image
        # original_img = Image.open(img_path).convert('RGB')
        original_img = gdal.Open(img_path).ReadAsArray()
        #!添加噪声
        # noise = np.random.normal(0, 2, original_img.shape)
        original_img = ((original_img )/255).astype(np.float32)
        if original_img.shape[1]<100 or original_img.shape[2]<100:
            continue
        original_img = original_img.transpose(1, 2, 0)
        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 6, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            #通过output输出概率
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            # 将不敢兴趣的区域像素设置成0(黑色)
            # prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            # mask.save(f"D:/agriculture-road-extraction/code/datasets/pred-dlinkdscsen/{img_path1[:-4]}.png")
            mask.save(f"{out_path}/{img_path1[:-4]}.png",dpi=(500,500))

if __name__ == '__main__':
    main()
