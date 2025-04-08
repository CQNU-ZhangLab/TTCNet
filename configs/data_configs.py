def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class mit():
    def __init__(self):
        super(mit, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['N', 'S', 'V', 'F', 'Q']
        self.sequence_len = 187

        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128

        # Transformer
        self.trans_dim = 25
        self.num_heads = 5


class ptb():
    def __init__(self):
        super(ptb, self).__init__()
        # data parameters
        self.num_classes = 2
        self.class_names = ['normal', 'abnormal']
        self.sequence_len = 188

        # model configs
        self.input_channels = 1  # 15
        self.kernel_size = 8
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128

        # Transformer
        self.trans_dim = 25
        self.num_heads = 5


class py20172():
    def __init__(self):
        super(py20172, self).__init__()
        # 数据参数
        self.num_classes = 4  # A -> 0, N -> 1, O -> 2, ~ -> 3
        self.class_names = ['A', 'N', 'O', '~']  # 类别名称
        self.sequence_len = 185  # 模型输入固定长度（可截断或零填充到3000点）

        # 模型配置
        self.input_channels = 1  # 单通道信号
        self.kernel_size = 8  # 卷积核大小
        self.stride = 1  # 卷积步长
        self.dropout = 0.2  # dropout概率

        # 特征参数
        self.mid_channels = 32  # 中间层通道数
        self.final_out_channels = 128  # 最终输出特征通道数

        # Transformer配置
        self.trans_dim = 25  # Transformer输入维度
        self.num_heads = 5  # Transformer头数