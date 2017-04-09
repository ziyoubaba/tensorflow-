#-*- coding:utf-8 -*-
import tensorflow as tf

# 配置神经网络的参数
input_node = 784
output_node = 10

image_size = 28
num_channels = 1
num_labels = 10

# 第一层卷积的尺度和深度
conv1_deep = 32
conv1_size = 5


# 第二层卷积的尺度和深度
conv2_deep = 64
conv2_size = 5

# 全连接层的节点数
fc_size = 512

# 定义卷积神经网络的前向传播过程。这里添加一个新的参数train ，用于区分训练过程和测试过程
# 这个过程将会用到dropout方法，可以进一步提高模型的可靠性病防止过拟合
#dropout 只在训练过程之中使用

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'): # 作用域（隔离变量到当前的）
        # 声明权重
        conv1_weight = tf.get_variable("weight",[conv1_size,conv1_size,num_channels,conv1_deep],
                                       initializer = tf.truncated_normal_initializer(stddev=0.1))
        # 声明偏移
        conv1_biases = tf.get_variable('bias',[conv1_deep] , initializer= tf.constant_initializer(0.0))
        # 使用边长为5 ，深度为32的过滤器，过滤器移动的步长为1 ，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor,conv1_weight , strides= [1,1,1,1] ,padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    # 使用第二层池化层的前向传播过程，这里选用最大池化层 池化层过滤器的边长为2
    # 使用全0填充 并且移动的步长为2。这一层的输入是上一层的输出，也就是说，是 28 × 28 * 32　的矩阵
    # 输出为　14*14 * 32的矩阵
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1 , ksize=[1,2,2,1] , strides= [1,2,2,1] ,padding='SAME')

    # 声明第三层卷基层的变量　并实现前向传播过程。这一层的输入为14*14*32的矩阵
    # 输出为　１４　×　１４　×　６４　的矩阵
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable("weight",[conv2_size , conv2_size , conv1_deep ,conv2_deep],
                                       initializer= tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[conv2_deep],initializer=tf.constant_initializer(0.0))
        # 使用边长为５，深度为６４的过滤器，过滤器移动的步长为１，且使用全０填充
        conv2 = tf.nn.conv2d(pool1 , conv1_weight ,strides=[1,1,1,1],padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    # 实现第四层池化层的前向传播网络。这一层和第二层的结构是一样的。
    # 这一层的输入为１４＊１４＊６４的矩阵，输出为７＊７＊６４的矩阵
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2 , ksize=[1,2,2,1] , strides=[1,2,2,1] ,padding="SAME")

    # 将第四层池化层的输出转化为第五层全连接的输入
    # 第四层的输出为　７＊７＊６４　的矩阵，然而第五层全连接层需要的输入格式为向量，所以在这里需要将这个７＊７＊６４的矩阵拉直成为一个向量
    # pool2.get_shape 函数可以得到第四层输出矩阵的维度而不需要手工计算。
    #　注意因为每一层神经网络的输入输出都为一个ｂａｔｃｈ的矩阵，所以这里得到的维度也包含了一个ｂａｔｃｈ中数据的个数
    pool_shape = pool2.get_shape().as_list()    # 得到ｐｏｏｌ２输入层的具体尺寸　[元素的个数,7,7,64]
    # 计算将矩阵拉直成为向量之后的长度，这个长度就是矩阵的长度及深度的乘积，注意这里
    # pool_shape[0]为一个batch中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 通过tf.reshape()函数将第四层的输出变成一个batch的向量
    reshaped =tf.reshape(pool2,[pool_shape[0],nodes])

    # 声明第五层全连接层的变量，并且实现全向传播过程．这一层的输入是拉直之后的一组向量．
    # 向量的长度为３１３６，输出是一组５１２的向量
    # 引入dropout的概念,dropout在训练时会随即将部分节点的输出改为0,可以避免过拟合的问题.从而使模型在测试数据上的效果更好
    # dropout 一般只在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,fc_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer!= None:
            tf.add_to_collection('losses',regularizer(fc1_weights)) # 为什么权重需要正则化?这里不理解
        fc1_biases = tf.get_variable('bias',[fc_size],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped , fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    # 声明第六层全连接层的变量,并且实现前向传播过程.这一层的输入为一组长度为512的向量
    # 输出为一组长度为10的向量,这一层的输出通过softmax之后就得到了最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weight = tf.get_variable("weight",[fc_size,num_labels],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weight))
        fc2_biases = tf.get_variable('bias',[num_channels],
                                     initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weight) + fc2_biases

    # 返回第六层的输出
    return logit