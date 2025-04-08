import torch
import torch.nn.functional as F
import numpy as np
import os
from models import ecgTransForm  # 假设你的模型路径
from dataloader import data_generator
from utils import _calc_metrics, to_device


class Tester(object):
    def __init__(self, args):
        # dataset parameters
        self.dataset = args.dataset
        self.device = torch.device(args.device)

        # paths
        self.home_path = os.getcwd()
        self.data_path = args.data_path
        self.checkpoint_dir = args.checkpoint_dir  # 用于加载模型的路径

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()
        self.hparams = self.hparams_class.train_params

    def get_configs(self):
        from configs.data_configs import get_dataset_class
        from configs.hparams import get_hparams_class

        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class("supervised")
        return dataset_class(), hparams_class()

    def load_data(self, data_type):
        self.test_dl, _, _, _ = data_generator(self.data_path, data_type, self.hparams)

    def load_model(self):
        model = ecgTransForm(configs=self.dataset_configs, hparams=self.hparams)
        model.to(self.device)

        # 加载训练好的模型
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")  # 这里根据你的文件名调整

        # 直接加载模型权重
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()  # 设置为评估模式
        return model

    def calc_results(self):
        acc, f1 = _calc_metrics(self.pred_labels, self.true_labels, self.dataset_configs.class_names)
        return acc, f1

    def test(self):
        # 加载数据
        self.load_data('py2017')

        # 加载模型
        model = self.load_model()

        total_loss_ = []
        self.pred_labels = np.array([])
        self.true_labels = np.array([])

        # 评估模式下运行
        with torch.no_grad():
            for batches in self.test_dl:
                batches = to_device(batches, self.device)
                data = batches['samples'].float()
                labels = batches['labels'].long()

                # forward pass
                predictions = model(data)

                # 计算损失
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())

                # 获取预测标签
                pred = predictions.detach().argmax(dim=1)  # 获取最大概率的类别索引

                # 收集所有预测和真实标签
                self.pred_labels = np.append(self.pred_labels, pred.cpu().numpy())
                self.true_labels = np.append(self.true_labels, labels.data.cpu().numpy())

        # 计算平均损失
        avg_loss = torch.tensor(total_loss_).mean()

        # 计算准确率和 F1 分数
        acc, f1 = self.calc_results()

        # 打印最终测试结果
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1-Score: {f1:.4f}")

        # 你可以将测试结果保存到文件或日志中


if __name__ == "__main__":
    # 假设你有命令行参数来传递给程序，这里以示例的方式传入
    class Args:
        dataset = 'py2017'
        device = 'cuda'  # 或者 'cpu'
        data_path = r'D:\Na\dataset'
        checkpoint_dir = r'D:\Na\GT\dmo5-py2017\experiments_logs\py2017-PY2017-epoch-200\run1_17_13'  # 请根据实际情况修改


    args = Args()

    tester = Tester(args)
    tester.test()
