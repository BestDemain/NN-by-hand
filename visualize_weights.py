import numpy as np
import matplotlib.pyplot as plt
import os
from model import ThreeLayerNet

def visualize_weights(model, layer='W1', num_neurons=25, save_path='results'):
    """
    可视化模型权重
    
    Args:
        model: 训练好的神经网络模型
        layer: 要可视化的层名称
        num_neurons: 要可视化的神经元数量
        save_path: 保存结果的路径
    """
    os.makedirs(save_path, exist_ok=True)
    weights = model.params[layer]
    
    if layer == 'W1':
        # 对于第一层权重，可以将其重塑为图像形状
        n_neurons = min(num_neurons, weights.shape[1])
        plt.figure(figsize=(10, 10))
        
        for i in range(n_neurons):
            plt.subplot(5, 5, i + 1)
            # 重塑为32x32x3的图像
            w = weights[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
            # 归一化权重以便可视化
            w = (w - w.min()) / (w.max() - w.min())
            plt.imshow(w)
            plt.axis('off')
            plt.title(f'Neuron {i+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weight_visualization.png'))
        plt.show()

def visualize_weight_distributions(model, save_path='results'):
    """
    可视化各层权重的分布
    
    Args:
        model: 训练好的神经网络模型
        save_path: 保存结果的路径
    """
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    # 可视化第一层权重分布
    plt.subplot(1, 3, 1)
    plt.hist(model.params['W1'].flatten(), bins=50)
    plt.title('W1 Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    # 可视化第二层权重分布
    plt.subplot(1, 3, 2)
    plt.hist(model.params['W2'].flatten(), bins=50)
    plt.title('W2 Distribution')
    plt.xlabel('Weight Value')
    
    # 可视化第三层权重分布
    plt.subplot(1, 3, 3)
    plt.hist(model.params['W3'].flatten(), bins=50)
    plt.title('W3 Distribution')
    plt.xlabel('Weight Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'weight_distributions.png'))
    plt.show()

def visualize_training_history(history, save_path='results'):
    """
    可视化训练历史
    
    Args:
        history: 包含训练和验证损失、准确率的字典
        save_path: 保存结果的路径
    """
    os.makedirs(save_path, exist_ok=True)
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
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.show()

if __name__ == "__main__":
    # 加载训练好的模型
    from config import DEFAULT_CONFIG, CIFAR10_CLASSES
    from utils import load_cifar10, preprocess_data
    
    # 加载数据集
    data_dir = 'data'
    X_train, y_train, X_test, y_test = load_cifar10(os.path.join(data_dir, 'cifar-10-batches-py'))
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    # 创建模型实例
    model = ThreeLayerNet(
        input_size=X_train.shape[1],
        hidden_size1=DEFAULT_CONFIG['hidden_size1'],
        hidden_size2=DEFAULT_CONFIG['hidden_size2'],
        output_size=len(CIFAR10_CLASSES),
        hidden_activation=DEFAULT_CONFIG['hidden_activation'],
        output_activation='softmax'
    )
    
    # 加载训练好的模型参数
    model_dir = DEFAULT_CONFIG['model_dir']
    model.load_model(os.path.join(model_dir, 'best_model.npz'))
    
    # 可视化权重
    visualize_weights(model)
    visualize_weight_distributions(model)
    
    # 如果有训练历史数据，也可以可视化
    # 这里需要加载训练历史数据，或者重新训练模型获取历史数据
    # visualize_training_history(history)
    
    print("权重可视化完成，结果保存在results目录下")