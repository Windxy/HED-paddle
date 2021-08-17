import paddle
import paddle.nn.functional as F

input = paddle.to_tensor([[1.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 0.0, 1.0]], dtype='float32')
label = paddle.to_tensor([[1.0, 0, 1.0],[1.0, 0.0, 1.0],[1.0, 0.0, 1.0]], dtype='float32')

def loss_func1(pre,y):
    num_positive = paddle.sum(y).numpy()                   # groud truth为1的个数
    num_negative = y.shape[-1]*y.shape[-2] - num_positive  # 宽和高，就是总的元素个数
    mask = paddle.assign(y)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    ans = F.binary_cross_entropy(pre,y,weight=mask)
    print(ans.numpy())
    return ans

def loss_func(pre,y):
    num_positive = paddle.sum(y)                        # groud truth为1的个数
    num_all      = paddle.to_tensor(y.shape[-1]*y.shape[-2],dtype=paddle.float32)                  # 宽和高，就是总的元素个数
    num_negative = paddle.to_tensor(num_all - num_positive,dtype=paddle.float32)
    weight_pos   = num_negative / (num_all + 1e-6)        # 论文中所属，正例权重为|Y-|/|Y|
    weight_neg   = num_positive / (num_all + 1e-6)
    weight       = weight_pos * y + weight_neg * (1 - y)      # 为1的label，给weight_pos，为0的label，给weight_neg
    pos_weight = paddle.to_tensor(num_negative/(num_all),dtype=paddle.float32)
    loss = paddle.nn.functional.binary_cross_entropy_with_logits(pre,y,
        weight = weight,pos_weight=pos_weight)
    # print(loss,weight_neg,weight_pos,weight)
    return loss

print(loss_func(input,label))
# output = F.binary_cross_entropy(input, label)
# print(output.numpy()) # [0.65537095]
