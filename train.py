import paddle
from datasets import HED_Dataset
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.vision.transforms import Compose,ToTensor
from net.hed import HED
import warnings
warnings.filterwarnings('ignore')


'''0.参数'''
epochs = 1
fre_print = 100
batch_size = 10

'''1.加载数据集'''
train_dataset = HED_Dataset(mode='train',transform=Compose([ToTensor()]))
# test_dataset = HED_Dataset(mode='test',transform=Compose([ToTensor()]))
train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)

'''2.搭建网络'''
net = HED()

'''3.损失函数'''
# 根据论文中定义的损失函数，每个branch和fuse与groud truth计算二值交叉熵loss，然后求和
# 下面这个函数是单个branch和gt求BCELoss，参考：
# Out=−1∗weight∗(label∗log(input)+(1−label)∗log(1−input))
# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/binary_cross_entropy_cn.html#binary-cross-entropy
def loss_func(pre,y):
    num_positive = paddle.sum(y)                        # groud truth为1的个数
    num_all      = paddle.to_tensor(y.shape[-1]*y.shape[-2],dtype=paddle.float32)                  # 宽和高，就是总的元素个数
    num_negative = paddle.to_tensor(num_all - num_positive,dtype=paddle.float32)
    weight_pos   = num_negative / (num_all + 1e-6)        # 论文中所属，正例权重为|Y-|/|Y|
    weight_neg   = num_positive / (num_all + 1e-6)
    weight       = weight_pos * y + weight_neg * (1 - y)      # 为1的label，给weight_pos，为0的label，给weight_neg
    pos_weight = paddle.to_tensor(num_negative/(num_all),dtype=paddle.float32)
    loss = paddle.nn.functional.binary_cross_entropy_with_logits(pre,y,
        weight = weight,reduction='sum',pos_weight=pos_weight)
    # print(loss,weight_neg,weight_pos,weight)
    return loss

def cross_entropy_loss(prediction, label):
    mask = (label != 0).float()
    num_positive = paddle.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = paddle.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask)
    return paddle.sum(cost)

'''4.优化器'''
scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.01, factor=0.5, patience=8, verbose=True)
adam = paddle.optimizer.Adam(learning_rate = scheduler,parameters=net.parameters())

'''5.训练'''
def train():
    net.train()
    for epoch in range(epochs):
        all_loss = 0
        counter = 0
        for batch_id,data in enumerate(train_loader):
            x = data[0]
            y = data[1]
            predict = net(x)
            loss = paddle.to_tensor(0,dtype=paddle.float32)
            for each_pre in predict:
                loss += loss_func(each_pre, y)
            loss.backward()

            all_loss += loss.numpy()
            counter+=1

            if batch_id % fre_print == 0:
                print("epoch: {}，batch_id: {},loss: {}".format(epoch,batch_id,all_loss/batch_size))
                all_loss = 0
            if counter == batch_size:    # 批次
                counter = 0
                adam.step()
                adam.clear_grad()
                scheduler.step(loss)

    '''保存'''
    paddle.save(net.state_dict(),'net.pdparams')
    paddle.save(adam.state_dict(),'adam.pdopt')

if __name__ == '__main__':
    train()


