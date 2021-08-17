import paddle
from net.hed import HED
import cv2
import numpy as np
from paddle.vision.transforms import Compose,ToTensor

'''1.加载模型'''
net = HED()
layer_state_dict = paddle.load('2_net.pdparams',)
net.set_state_dict(layer_state_dict)
net.train()

'''2.加载图片'''
img_file = 'HED-BSDS\\test\\8068.jpg'
or_img = cv2.imread(img_file)
# img = or_img[np.newaxis,:,:,:].astype(np.float32)
# img = paddle.to_tensor(img)
transform = Compose([ToTensor()])
img = np.array(or_img, dtype=np.float32)
img = transform(img)
img = paddle.unsqueeze(img,0)
# img = paddle.transpose(img,[0,3,1,2])
y = net(img)
for each_layer in y:
    e = paddle.squeeze(each_layer)
    e = e.numpy()
    e = e*255
    cv2.imshow('1',e)
    cv2.imshow('ori',or_img)
    cv2.waitKey(0)