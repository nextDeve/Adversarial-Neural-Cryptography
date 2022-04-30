#### 字段说明
<ol>
    <li>max_epoch：最大训练轮次</li>
    <li>plain：明文的位数</li>
    <li>key：秘钥的位数</li>
    <li>cipher：密文的位数</li>
    <li>batch_size：每个step生成的明文和密钥组数</li>
    <li>device：训练设备</li>
    <li>lr：模型学习率</li>
    <li>step：steps[0] 表示每个epoch 训练集生成多少组明文和密钥，steps[1] 表示每个epoch 测试集生成多少组明文和密钥</li>
    <li>alice：
        <ol>
            <li>depth：alice的网络深度</li>
            <li>hidden：alice 每层神经单元的个数</li>
        </ol>
    </li>
    <li>bob：
        <ol>
            <li>depth：bob的网络深度</li>
            <li>hidden：bob 每层神经单元的个数</li>
        </ol>
    </li>
    <li>eve：
        <ol>
            <li>depth：eve的网络深度</li>
            <li>hidden：eve 每层神经单元的个数</li>
        </ol>
    </li>
    <li>loss_path:存放模型训练过程损失变化文件目录</li>
    <li>model_path:存放模型文件目录</li>
</ol>

#### 文件说明
~~~~
./img:存放图片
./model_dict    存放训练好的模型
./loss_data     存放模型训练损失数据
dataset.py      生成训练和测试数据集
model.py        定义模型结构
train.py        训练过程
validate.py     测试过程
main.py         程序入口
predict.py      加密解密样例
plot.py         画损失图和正确率
~~~~
#### 程序执行顺序

<ul>
    <li>
    运行main.py：python main.py
    <p>等待命令执行完毕，模型训练完成</p>
    </li>
    <li>
     运行plot.py：python plot.py
     <p>画损失图和正确率图</p>
    </li>
     <li>
     运行predict.py：python predict.py
     <p>加解密测试</p>
    </li>
</ul>

#### 程序依赖库
~~~~
torch
tqdm
pandas
numpy
~~~~
