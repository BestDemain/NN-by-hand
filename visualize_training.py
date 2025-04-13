import numpy as np
import matplotlib.pyplot as plt
import os
import json
from model import ThreeLayerNet
from config import DEFAULT_CONFIG, CIFAR10_CLASSES
from utils import load_cifar10, preprocess_data, plot_training_history
from train import train_model

def load_search_results(file_path='search_results/search_results.json'):
    """
    加载超参数搜索结果
    
    Args:
        file_path: 搜索结果文件路径
        
    Returns:
        搜索结果字典
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_training_curves(save_path='results'):
    """
    生成训练曲线
    
    Args:
        save_path: 保存结果的路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据集
    data_dir = 'data'
    X_train, y_train, X_test, y_test = load_cifar10(os.path.join(data_dir, 'cifar-10-batches-py'))
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    # 创建配置
    config = DEFAULT_CONFIG.copy()
    config.update({
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'epochs': 20  # 使用较少的轮数以加快训练
    })
    
    # 训练模型并获取历史记录
    print("训练模型以获取历史记录...")
    model, history = train_model(config)
    
    # 绘制训练历史
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
    plt.close()
    
    print(f"训练曲线已保存到 {os.path.join(save_path, 'training_history.png')}")
    
    return history

def visualize_best_model_performance(save_path='results'):
    """
    可视化最佳模型在各个类别上的性能
    
    Args:
        save_path: 保存结果的路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据集
    data_dir = 'data'
    X_train, y_train, X_test, y_test = load_cifar10(os.path.join(data_dir, 'cifar-10-batches-py'))
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    # 加载最佳模型
    model = ThreeLayerNet(
        input_size=X_train.shape[1],
        hidden_size1=DEFAULT_CONFIG['hidden_size1'],
        hidden_size2=DEFAULT_CONFIG['hidden_size2'],
        output_size=len(CIFAR10_CLASSES),
        hidden_activation=DEFAULT_CONFIG['hidden_activation'],
        output_activation='softmax'
    )
    model_dir = DEFAULT_CONFIG['model_dir']
    model.load_model(os.path.join(model_dir, 'best_model.npz'))
    
    # 获取测试集预测结果
    y_pred = model.predict(X_test)
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(len(CIFAR10_CLASSES)):
        # 找出属于该类别的样本
        idx = (y_test == i)
        if np.sum(idx) > 0:  # 确保有该类别的样本
            # 计算该类别的准确率
            acc = np.mean(y_pred[idx] == y_test[idx])
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    # 绘制每个类别的准确率
    plt.figure(figsize=(10, 6))
    plt.bar(CIFAR10_CLASSES, class_accuracies)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_accuracy.png'))
    plt.close()
    
    print(f"类别准确率图已保存到 {os.path.join(save_path, 'class_accuracy.png')}")

if __name__ == "__main__":
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 生成训练曲线
    history = generate_training_curves()
    
    # 可视化最佳模型在各个类别上的性能
    visualize_best_model_performance()
    
    print("训练过程可视化完成！")