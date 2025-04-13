import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_cifar10_batch(file_path):
    """
    加载CIFAR-10数据集的一个批次
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        X: 图像数据，形状为(10000, 3072)
        y: 标签，形状为(10000,)
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    # 提取图像和标签
    X = batch[b'data']
    y = batch[b'labels']
    
    return X, np.array(y)

def load_cifar10(data_dir):
    """
    加载CIFAR-10数据集
    
    Args:
        data_dir: 数据集目录路径
        
    Returns:
        X_train: 训练图像数据，形状为(50000, 3072)
        y_train: 训练标签，形状为(50000,)
        X_test: 测试图像数据，形状为(10000, 3072)
        y_test: 测试标签，形状为(10000,)
    """
    # 加载训练数据
    X_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        X_batch, y_batch = load_cifar10_batch(batch_file)
        X_train.append(X_batch)
        y_train.append(y_batch)
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    # 加载测试数据
    test_file = os.path.join(data_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_file)
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test, validation_split=0.1):
    """
    预处理CIFAR-10数据集
    
    Args:
        X_train: 训练图像数据，形状为(N, 3072)
        y_train: 训练标签，形状为(N,)
        X_test: 测试图像数据，形状为(M, 3072)
        y_test: 测试标签，形状为(M,)
        validation_split: 验证集比例
        
    Returns:
        处理后的训练集、验证集和测试集
    """
    # 将像素值归一化到[0, 1]范围
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # 标准化（零均值，单位方差）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 分割训练集和验证集
    if validation_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    return X_train, y_train, X_test, y_test

def get_mini_batches(X, y, batch_size):
    """
    生成小批量数据
    
    Args:
        X: 输入数据
        y: 标签
        batch_size: 批量大小
        
    Returns:
        小批量数据生成器
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]

def visualize_sample(X, y, class_names=None):
    """
    可视化CIFAR-10样本
    
    Args:
        X: 图像数据，形状为(N, 3072)
        y: 标签，形状为(N,)
        class_names: 类别名称列表
    """
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 将图像数据重塑为(N, 32, 32, 3)
    X_reshaped = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # 如果数据已经标准化，需要恢复到[0, 1]范围
    if X_reshaped.min() < 0:
        X_reshaped = (X_reshaped - X_reshaped.min()) / (X_reshaped.max() - X_reshaped.min())
    
    # 显示图像
    plt.figure(figsize=(10, 10))
    for i in range(min(25, X.shape[0])):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_reshaped[i])
        plt.title(class_names[y[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    绘制训练历史
    
    Args:
        history: 包含训练和验证损失、准确率的字典
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    plt.tight_layout()
    plt.show()

def calculate_accuracy(y_pred, y_true):
    """
    计算分类准确率
    
    Args:
        y_pred: 预测的类别索引，形状为(N,)
        y_true: 真实的类别索引，形状为(N,)
        
    Returns:
        准确率
    """
    return np.mean(y_pred == y_true)

def download_cifar10(download_dir):
    """
    下载CIFAR-10数据集
    
    Args:
        download_dir: 下载目录
    """
    import tarfile
    import urllib.request
    
    # 创建下载目录
    os.makedirs(download_dir, exist_ok=True)
    
    # 下载数据集
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_path = os.path.join(download_dir, "cifar-10-python.tar.gz")
    
    if not os.path.exists(file_path):
        print("正在下载CIFAR-10数据集...")
        urllib.request.urlretrieve(url, file_path)
        print("下载完成！")
    
    # 解压数据集
    extract_dir = os.path.join(download_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        print("正在解压数据集...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print("解压完成！")
    
    return extract_dir