import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, '{0}/app'.format(os.path.dirname(__file__)))
import RRDBNet_arch as arch

# Settings
model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')
test_img_folder = 'LR/*'

# ESRGAN Model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# Supported Extensions
img_ext = ['.bmp','.dib','.jpeg','.jpg','.jpe','.jp2','.png','.pbm','.pgm','.ppm','.sr','.ras','.tiff','.tif']
vid_ext = ['.mp4']

def ResizeImage(img):
    # Resize image to have a dimension less than 100 pixels
    height, width = img.shape[0], img.shape[1]
    scale_factor = 100 / (height if (height > width) else width)
    dim = (int(width*scale_factor), int(height*scale_factor))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def ESRGAN(img, model):
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output


print('Model path {:s}.'.format(model_path))
for path in glob.glob(test_img_folder):
    base, ext = osp.splitext(osp.basename(path))
    print('{0}{1}'.format(base, ext))

    if (ext in img_ext):
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if (100 not in img.shape):
            img = ResizeImage(img, base, ext)
            cv2.imwrite('LR/{:s}{:s}'.format(base, ext), img)
        output = ESRGAN(img, model)
        cv2.imwrite('results/{:s}.png'.format(base), output)
    elif (ext in vid_ext):
        # Read frames
        filename = "results/{:s}.avi".format(base)

        cap = cv2.VideoCapture(path)
        writer = None
        if not cap.isOpened():
            exit()
        ret, frame = cap.read()

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame if (100 in frame.shape) else ResizeImage(frame)
            frame = ESRGAN(frame, model)
            
            if (writer == None):
                codec = cv2.VideoWriter_fourcc(*'DIVX')
                framerate = cap.get(cv2.CAP_PROP_FPS)
                resolution = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(filename, codec, framerate, resolution)

            writer.write(frame.astype('uint8'))
            
        writer.release()
        cap.release()