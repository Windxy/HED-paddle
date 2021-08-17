import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import numpy as np

class HED(nn.Layer):
    '''
    HED网络结构为VGG16删去后面的全连接层和softmax层
    包括5个stage：
    stage1由2个卷积和1个MaxPooling构成，每个卷积后接一个ReLU
    stage2由3个卷积和1个MaxPooling构成，每个卷积后接一个ReLU
    stage3由3个卷积和1个MaxPooling构成，每个卷积后接一个ReLU
    stage4由3个卷积和1个MaxPooling构成，每个卷积后接一个ReLU
    stage5由3个卷积和1个MaxPooling构成，每个卷积后接一个ReLU
    '''
    def __init__(self):
        super(HED, self).__init__()
        '''stage1'''
        self.conv1_1 = nn.Conv2D(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2D(64, 64, 3, padding=1)

        '''stage2'''
        self.conv2_1 = nn.Conv2D(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2D(128, 128, 3, padding=1)

        '''stage3'''
        self.conv3_1 = nn.Conv2D(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2D(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2D(256, 256, 3, padding=1)

        '''stage4'''
        self.conv4_1 = nn.Conv2D(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2D(512, 512, 3, padding=1)

        '''stage5'''
        self.conv5_1 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2D(512, 512, 3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.branch1 = nn.Conv2D(64, 1, 1)
        self.branch2 = nn.Conv2D(128, 1, 1)
        self.branch3 = nn.Conv2D(256, 1, 1)
        self.branch4 = nn.Conv2D(512, 1, 1)
        self.branch5 = nn.Conv2D(512, 1, 1)
        self.score_final = nn.Conv2D(5, 1, 1)

        self.sigmoid = nn.Sigmoid()

        # self.deconv2 = nn.Conv2DTranspose(1,1,3,2)
        # self.deconv3 = nn.Conv2DTranspose(1,1,4,4)
        # self.deconv4 = nn.Conv2DTranspose(1,1,8,8)
        # self.deconv5 = nn.Conv2DTranspose(1,1,16,16)

        # self.proconv2 = nn.Conv2D(1,1,3,1,1)
        # self.proconv3 = nn.Conv2D(1,1,3,1,1)
        # self.proconv4 = nn.Conv2D(1,1,3,1,1)
        # self.proconv5 = nn.Conv2D(1,1,3,1,1)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)

    def forward(self,x):
        '''
        :param x.shape:(Batch_size,Channel,Height,Weight)
        :return: y.shape:(Batch_size,1,Height,Weight)
        '''
        img_H, img_W = x.shape[2], x.shape[3]
        '''VGG16的前13个阶段'''
        '''stage1'''
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))  # batch_size,64,H,W
        pool1 = self.maxpool(conv1_2)

        '''stage2'''
        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))  # batch_size,128,H/2,W/2
        pool2 = self.maxpool(conv2_2)

        '''stage3'''
        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))  # batch_size,256,H/4,W/4
        pool3 = self.maxpool(conv3_3)

        '''stage4'''
        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))  # batch_size,512,H/8,W/8
        pool4 = self.maxpool(conv4_3)

        '''stage5'''
        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))  # batch_size,512,H/16,W/16 (14)

        '''每个stage对应的分支'''
        # b1 = self.relu(self.branch1(conv1_2))  # batch_size,1,H,W
        # b2 = self.relu(self.branch2(conv2_2))  # batch_size,1,H/2,W/2
        # b3 = self.relu(self.branch3(conv3_3))  # batch_size,1,H/4,W/4
        # b4 = self.relu(self.branch4(conv4_3))  # batch_size,1,H/8,W/8
        # b5 = self.relu(self.branch5(conv5_3))  # batch_size,1,H/16,W/16
        b1 = self.branch1(conv1_2)  # batch_size,1,H,W
        b2 = self.branch2(conv2_2)  # batch_size,1,H/2,W/2
        b3 = self.branch3(conv3_3)  # batch_size,1,H/4,W/4
        b4 = self.branch4(conv4_3)  # batch_size,1,H/8,W/8
        b5 = self.branch5(conv5_3)  # batch_size,1,H/16,W/16

        '''每个分支进行线性插入（上采样），第一个分支不用，因为其就是原图的长宽'''
        b2 = F.interpolate(b2, size=[img_H,img_W], mode='bilinear')      # 也可使用nn.functional.conv2d_transpose转置卷积
        b3 = F.interpolate(b3, size=[img_H,img_W], mode='bilinear')
        b4 = F.interpolate(b4, size=[img_H,img_W], mode='bilinear')
        b5 = F.interpolate(b5, size=[img_H,img_W], mode='bilinear')
        # b2 = F.conv2d_transpose()
        # b2 = self.deconv2(b2)
        # b3 = self.deconv3(b3)
        # b4 = self.deconv4(b4)
        # b5 = self.deconv5(b5)

        # print(b2.shape,b3.shape,b4.shape,b5.shape)
        # print(x.shape[0])
        b2 = paddle.crop(b2,shape=(1,1,img_H,img_W))
        b3 = paddle.crop(b3,shape=(1,1,img_H,img_W))
        b4 = paddle.crop(b4,shape=(1,1,img_H,img_W))
        b5 = paddle.crop(b5,shape=(1,1,img_H,img_W))

        # b2 = self.proconv2(b2)
        # b3 = self.proconv2(b3)
        # b4 = self.proconv2(b4)
        # b5 = self.proconv2(b5)

        '''连接每一个分支'''
        fusecat = paddle.concat([b1, b2, b3, b4, b5], axis=1)
        fuse = self.score_final(fusecat)
        results = [b1, b2, b3, b4, b5, fuse]
        results = [self.sigmoid(r) for r in results]
        return results

if __name__ == '__main__':
    net = HED()
    x = paddle.rand((1,3,224,224))
    y = net(x)
    for i in y:
        print(i.shape)
