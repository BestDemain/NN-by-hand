# 默认配置参数

DEFAULT_CONFIG = {
    # 模型参数
    'input_size': 3072,  # CIFAR-10: 32x32x3 = 3072
    'hidden_size1': 512,
    'hidden_size2': 256,
    'output_size': 10,  # CIFAR-10: 10个类别
    'hidden_activation': 'relu',
    'output_activation': 'softmax',
    
    # 训练参数
    'loss': 'cross_entropy',
    'optimizer': 'sgd',
    'learning_rate': 0.2,
    'momentum': 0.9,
    'weight_decay': 0.0001,  # L2正则化系数
    'batch_size': 256,
    'epochs': 25,
    
    # 学习率调度
    'scheduler': 'step',
    'scheduler_params': {
        'step_size': 1,
        'gamma': 0.98
    },
    
    # 数据参数
    'validation_split': 0.1,  # 验证集比例
    
    # 路径参数
    'data_dir': 'data',
    'model_dir': 'models',
    'results_dir': 'results'
}

# 超参数搜索配置
SEARCH_CONFIG = {
    # 网格搜索参数
    'grid_search': {
        'hidden_size1': [128, 256, 512],
        'hidden_size2': [64, 128, 256],
        'hidden_activation': ['relu', 'tanh'],
        'learning_rate': [0.1, 0.01, 0.001],
        'momentum': [0.0, 0.9],
        'weight_decay': [0.0, 0.0001, 0.001]
    },
    
    # 随机搜索参数
    'random_search': {
        'n_iter': 10,  # 随机搜索迭代次数
        'epochs': 10,  # 每次搜索的训练轮数
    },
    
    # 随机搜索分布
    'random_distributions': {
        'hidden_size1': [128, 256, 512, 1024],
        'hidden_size2': [64, 128, 256, 512],
        'hidden_activation': ['relu', 'tanh'],
        'learning_rate': 'log_uniform(-4, -1)',  # 对数均匀分布
        'momentum': 'uniform(0, 0.99)',
        'weight_decay': 'log_uniform(-5, -2)'  # 对数均匀分布
    }
}

# CIFAR-10数据集类别
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]