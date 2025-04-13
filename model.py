import numpy as np
from activations import get_activation

class ThreeLayerNet:
    """
    三层神经网络分类器
    
    结构：输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, 
                 hidden_activation='relu', output_activation='softmax'):
        """
        初始化三层神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_size1: 第一隐藏层神经元数量
            hidden_size2: 第二隐藏层神经元数量
            output_size: 输出类别数量
            hidden_activation: 隐藏层激活函数名称
            output_activation: 输出层激活函数名称
        """
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        # 获取激活函数
        self.hidden_activation = get_activation(hidden_activation)
        self.output_activation = get_activation(output_activation)
        
        # 初始化权重和偏置
        # Xavier/Glorot初始化，帮助解决梯度消失/爆炸问题
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
        self.params['b3'] = np.zeros(output_size)
        
        # 缓存中间结果，用于反向传播
        self.cache = {}
    
    def forward(self, X):
        """
        前向传播
        
        Args:
            X: 输入数据，形状为(batch_size, input_size)
            
        Returns:
            输出层的激活值，形状为(batch_size, output_size)
        """
        # 获取参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # 第一隐藏层
        z1 = np.dot(X, W1) + b1
        a1 = self.hidden_activation.forward(z1)
        
        # 第二隐藏层
        z2 = np.dot(a1, W2) + b2
        a2 = self.hidden_activation.forward(z2)
        
        # 输出层
        z3 = np.dot(a2, W3) + b3
        a3 = self.output_activation.forward(z3)
        
        # 缓存中间结果，用于反向传播
        self.cache = {
            'X': X,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3
        }
        
        return a3
    
    def backward(self, dout):
        """
        反向传播，计算梯度
        
        Args:
            dout: 输出层的梯度，形状为(batch_size, output_size)
            
        Returns:
            参数的梯度字典
        """
        # 获取参数
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        
        # 获取缓存的中间结果
        X = self.cache['X']
        z1, a1 = self.cache['z1'], self.cache['a1']
        z2, a2 = self.cache['z2'], self.cache['a2']
        z3 = self.cache['z3']
        
        # 计算梯度
        batch_size = X.shape[0]
        
        # 输出层的梯度（dout已经包含了softmax和交叉熵的组合梯度）
        # 如果不是组合梯度，则需要乘以输出激活函数的导数
        # dz3 = dout * self.output_activation.backward(z3)
        dz3 = dout
        
        # 第三层权重和偏置的梯度
        dW3 = np.dot(a2.T, dz3) / batch_size
        db3 = np.sum(dz3, axis=0) / batch_size
        
        # 第二隐藏层的梯度
        da2 = np.dot(dz3, W3.T)
        dz2 = da2 * self.hidden_activation.backward(z2)
        
        # 第二层权重和偏置的梯度
        dW2 = np.dot(a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0) / batch_size
        
        # 第一隐藏层的梯度
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * self.hidden_activation.backward(z1)
        
        # 第一层权重和偏置的梯度
        dW1 = np.dot(X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0) / batch_size
        
        # 返回梯度字典
        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }
        
        return grads
    
    def predict(self, X):
        """
        预测类别
        
        Args:
            X: 输入数据，形状为(batch_size, input_size)
            
        Returns:
            预测的类别索引，形状为(batch_size,)
        """
        # 前向传播
        a3 = self.forward(X)
        
        # 返回概率最大的类别索引
        return np.argmax(a3, axis=1)
    
    def save_model(self, file_path):
        """
        保存模型参数
        
        Args:
            file_path: 保存路径
        """
        np.savez(file_path, **self.params)
    
    def load_model(self, file_path):
        """
        加载模型参数
        
        Args:
            file_path: 模型参数文件路径
        """
        data = np.load(file_path)
        for key in self.params:
            if key in data:
                self.params[key] = data[key]