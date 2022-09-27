from Data_Process import *
from backbone import VIT
from sklearn.metrics import confusion_matrix
from PIL import Image
from torch.nn import functional as F
import torch.optim as optim
from libtiff import TIFF
import numpy as np
import torch
import cv2


# 1.定义网络超参数
EPOCH = 30  # 训练多少轮次
BATCH_SIZE = 48  # 每次喂给的数据量
LR = 0.001  # 学习率
Train_Rate = 0.1  # 将训练集和测试集按比例分开
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否用GPU环视cpu训练

# 读取图片——ms
ms4_tif = TIFF.open('../dataset/image1/ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()
print('原始ms4图的形状：', np.shape(ms4_np))

# 读取PAN图
pan_tif = TIFF.open('../dataset/image1/pan.tif', mode='r')
pan_np = pan_tif.read_image()
print('原始pan图的形状;', np.shape(pan_np))

# 读取标签
label_np = np.load("../dataset/image1/label4.npy")
print('label数组形状：', np.shape(label_np))

# ms4与pan图补零  (给图片加边框）
Ms4_patch_size = 16  # ms4截块的边长
Interpolation = cv2.BORDER_REFLECT_101
# cv2.BORDER_REPLICATE： 进行复制的补零操作;
# cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
# cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
# cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdefgh|abcdefgh|abcdefg;
#############I分量与HS分量相同操作##############
# [800,830, 4]->[815,845,4]  [3200,3320,1]->[3260, 3380, 1]
top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4图的形状：', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan图的形状：', np.shape(pan_np))

# 按类别比例拆分数据集
# label_np=label_np.astype(np.uint8)
label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255

label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
print('类标：', label_element)
print('各类样本数：', element_count)
Categories_Number = len(label_element) - 1  # 数据的类别数
print('标注的类别数：', Categories_Number)
label_row, label_column = np.shape(label_np)  # 获取标签图的行、列

'''归一化图片'''


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


ground_xy = np.array([[]] * Categories_Number).tolist()  # [[],[],[],[],[],[],[]]  7个类别
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)  # [800*830, 2] 二维数组

count = 0
for row in range(label_row):  # 行
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])  # 记录属于每个类别的位置集合

# 标签内打乱
for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)
    ground_xy[categories] = ground_xy[categories][shuffle_array]
shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    for i in range(categories_number):
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
        else:
            ground_xy_test.append(ground_xy[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
    label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

# 训练数据与测试数据，数据集内打乱
shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

print('训练样本数：', len(label_train))
print('测试样本数：', len(label_test))

# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)

pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)

train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 定义优化器
model = VIT().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0.0
    for step, (ms, pan, label, _) in enumerate(train_loader):
        ms, pan, label = ms.to(DEVICE), pan.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        output = model(ms, pan)
        pred_train = output[1].max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        # print('Train_cross_entropy', F.cross_entropy(output, label.long()).item())
        loss = F.cross_entropy(output[1], label.long()) + 0.01 * output[0]
        # 定义反向传播
        loss.backward()
        # 定义优化
        optimizer.step()
        train_loss = loss / len(train_loader.dataset)
        if step % 1000 == 0:
            print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(epoch, train_loss, step))
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))


# 定义测试方法
def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, data1, target, _ in test_loader:
            data, data1, target = data.to(DEVICE), data1.to(DEVICE), target.to(DEVICE)
            output = model(data, data1)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))


# 调用训练和测试
for epoch in range(1, EPOCH + 1):
    train_model(model, train_loader, optimizer, epoch)
    test_model(model, test_loader)
torch.save(model, '../model/Transformer5.pkl')

# 上色
transformer = torch.load('../model/Transformer5.pkl')
transformer.to(DEVICE)
class_count = np.zeros(7)
out_clour = np.zeros((800, 830, 3))


def clour_model(transformer, all_data_loader):
    for step, (ms4, pan, gt_xy) in enumerate(all_data_loader):
        ms4 = ms4.to(DEVICE)
        pan = pan.to(DEVICE)

        with torch.no_grad():
            output = transformer(ms4, pan)
        pred_y = output.data.max(1, keepdim=True)[1]
        pred_y_numpy = pred_y.cpu().numpy()
        gt_xy = gt_xy.numpy()
        for k in range(len(gt_xy)):
            if pred_y_numpy[k] == 0:
                class_count[0] = class_count[0] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [203, 192, 255]
            elif pred_y_numpy[k] == 1:
                class_count[1] = class_count[1] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [14, 132, 241]
            elif pred_y_numpy[k] == 2:
                class_count[2] = class_count[2] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
            elif pred_y_numpy[k] == 3:
                class_count[3] = class_count[3] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 255]
            elif pred_y_numpy[k] == 4:
                class_count[4] = class_count[4] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [51, 102, 103]
            elif pred_y_numpy[k] == 5:
                class_count[5] = class_count[5] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
            elif pred_y_numpy[k] == 6:
                class_count[6] = class_count[6] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
    print(class_count)
    picture = Image.fromarray(np.uint8(out_clour))
    picture.show()
    cv2.imwrite("xian_label.png", out_clour)


clour_model(model, all_data_loader)


# -------------------------------计算kappa指标---------------------------
def con_mat():
    transformer.to(DEVICE)
    l = 0
    y_pred = []
    for step, (ms4, pan, label, _) in enumerate(test_loader):
        l += 1
        ms4 = ms4.to(DEVICE)
        pan = pan.to(DEVICE)
        label = label.to(DEVICE)
        with torch.no_grad():
            output = transformer(ms4, pan)
        pred_y = output.max(1, keepdim=True)[1]
        if l == 1:
            y_pred = pred_y.cpu().numpy()
        else:
            y_pred = np.concatenate((y_pred, pred_y.cpu().numpy()), axis=0)

    con_mat = confusion_matrix(y_true=label_test, y_pred=y_pred)
    print("混淆矩阵", con_mat)

    # 计算性能参数
    all_acr = 0
    p = 0
    column = np.sum(con_mat, axis=0)  # 列求和
    line = np.sum(con_mat, axis=1)  # 行求和
    for i, clas in enumerate(con_mat):
        precise = clas[i]
        all_acr = precise + all_acr
        acr = precise / column[i]
        recall = precise / line[i]
        f1 = 2 * acr * recall / (acr + recall)
        temp = column[i] * line[i]
        p = p + temp
        # print('PRECISION:',acr,'||RECALL:',recall,'||F1:',f1)#查准率 #查全率 #F1
        print("第 %d 类: || 准确率: %.7f || 召回率: %.7f || F1: %.7f " % (i, acr, recall, f1))
    OA = np.trace(con_mat) / np.sum(con_mat)
    print('OA:', OA)

    AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))  # axis=1 每行求和
    print('AA:', AA)

    Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
    Kappa = (OA - Pc) / (1 - Pc)
    print('Kappa:', Kappa)


con_mat()
