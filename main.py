import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import ThreeLayerNet
from losses import get_loss
from optimizer import get_optimizer, get_scheduler
from train import train_model
from evaluation import load_and_test, evaluate_model_classes
from hyperparameter_search import HyperparameterSearch, example_grid_search, example_random_search
from utils import download_cifar10, load_cifar10, preprocess_data, visualize_sample, plot_training_history
from config import DEFAULT_CONFIG, SEARCH_CONFIG, CIFAR10_CLASSES

def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='手工实现三层神经网络分类器')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'search', 'visualize'],
                        help='运行模式：训练、测试、超参数搜索或可视化')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据集目录')
    parser.add_argument('--download', action='store_true',
                        help='是否下载CIFAR-10数据集')
    
    # 模型参数
    parser.add_argument('--hidden_size1', type=int, default=DEFAULT_CONFIG['hidden_size1'],
                        help='第一隐藏层大小')
    parser.add_argument('--hidden_size2', type=int, default=DEFAULT_CONFIG['hidden_size2'],
                        help='第二隐藏层大小')
    parser.add_argument('--hidden_activation', type=str, default=DEFAULT_CONFIG['hidden_activation'],
                        help='隐藏层激活函数')
    
    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=DEFAULT_CONFIG['momentum'],
                        help='动量系数')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_CONFIG['weight_decay'],
                        help='L2正则化系数')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='训练轮数')
    
    # 学习率调度参数
    parser.add_argument('--scheduler', type=str, default=DEFAULT_CONFIG['scheduler'],
                        help='学习率调度器类型')
    
    # 路径参数
    parser.add_argument('--model_dir', type=str, default=DEFAULT_CONFIG['model_dir'],
                        help='模型保存目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径（用于测试模式）')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_CONFIG['results_dir'],
                        help='结果保存目录')
    
    # 超参数搜索参数
    parser.add_argument('--search_type', type=str, default='grid', choices=['grid', 'random'],
                        help='超参数搜索类型：网格搜索或随机搜索')
    parser.add_argument('--search_epochs', type=int, default=10,
                        help='超参数搜索中每个配置的训练轮数')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 下载或加载CIFAR-10数据集
    cifar10_dir = os.path.join(args.data_dir, 'cifar-10-batches-py')
    if args.download or not os.path.exists(cifar10_dir):
        cifar10_dir = download_cifar10(args.data_dir)
    
    # 加载数据集
    try:
        X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)
        print(f"加载CIFAR-10数据集成功！训练集: {X_train.shape}, 测试集: {X_test.shape}")
    except Exception as e:
        print(f"加载CIFAR-10数据集失败: {e}")
        print("请确保数据集已下载，或使用--download参数下载数据集")
        return
    
    # 预处理数据
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test, validation_split=DEFAULT_CONFIG['validation_split']
    )
    print(f"数据预处理完成！训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        # 训练模式
        print("\n开始训练模型...")
        
        # 构建训练配置
        train_config = {
            'input_size': X_train.shape[1],
            'hidden_size1': args.hidden_size1,
            'hidden_size2': args.hidden_size2,
            'output_size': len(CIFAR10_CLASSES),
            'hidden_activation': args.hidden_activation,
            'output_activation': 'softmax',
            'loss': 'cross_entropy',
            'optimizer': 'sgd',
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'scheduler': args.scheduler,
            'scheduler_params': DEFAULT_CONFIG['scheduler_params'],
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'model_dir': args.model_dir,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }
        
        # 训练模型
        model, history = train_model(train_config)
        
        # 绘制训练历史
        plot_training_history(history)
        
        # 在测试集上评估
        print("\n在测试集上评估模型...")
        test_acc = load_and_test(
            os.path.join(args.model_dir, 'best_model.npz'),
            X_test, y_test,
            model_config={
                'input_size': X_train.shape[1],
                'hidden_size1': args.hidden_size1,
                'hidden_size2': args.hidden_size2,
                'output_size': len(CIFAR10_CLASSES),
                'hidden_activation': args.hidden_activation,
                'output_activation': 'softmax'
            }
        )
        
        # 评估各类别性能
        evaluate_model_classes(model, X_test, y_test, CIFAR10_CLASSES)
        
    elif args.mode == 'test':
        # 测试模式
        print("\n开始测试模型...")
        
        # 检查模型路径
        model_path = args.model_path
        if model_path is None:
            model_path = os.path.join(args.model_dir, 'best_model.npz')
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return
        
        # 加载并测试模型
        test_acc = load_and_test(
            model_path,
            X_test, y_test,
            model_config={
                'input_size': X_train.shape[1],
                'hidden_size1': args.hidden_size1,
                'hidden_size2': args.hidden_size2,
                'output_size': len(CIFAR10_CLASSES),
                'hidden_activation': args.hidden_activation,
                'output_activation': 'softmax'
            }
        )
        
        # 创建模型实例用于评估各类别性能
        model = ThreeLayerNet(
            input_size=X_train.shape[1],
            hidden_size1=args.hidden_size1,
            hidden_size2=args.hidden_size2,
            output_size=len(CIFAR10_CLASSES),
            hidden_activation=args.hidden_activation,
            output_activation='softmax'
        )
        model.load_model(model_path)
        
        # 评估各类别性能
        evaluate_model_classes(model, X_test, y_test, CIFAR10_CLASSES)
        
    elif args.mode == 'search':
        # 超参数搜索模式
        print("\n开始超参数搜索...")
        
        if args.search_type == 'grid':
            # 网格搜索
            print("执行网格搜索...")
            best_config, results = example_grid_search(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
        else:
            # 随机搜索
            print("执行随机搜索...")
            best_config, results = example_random_search(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
        
        # 打印最佳配置
        print("\n超参数搜索完成！")
        print("最佳配置:")
        for name, value in best_config.items():
            print(f"  {name}: {value}")
        
    elif args.mode == 'visualize':
        # 可视化模式
        print("\n可视化CIFAR-10样本...")
        
        # 随机选择一些样本进行可视化
        indices = np.random.choice(len(X_test), 25, replace=False)
        visualize_sample(X_test[indices], y_test[indices], CIFAR10_CLASSES)

if __name__ == '__main__':
    main()