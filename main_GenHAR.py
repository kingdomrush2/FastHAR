import ctypes

libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.FastHAR import TimeSeriesTransformer
import argparse
from utils.dataloader import  IMUDataset_fft
import os
import numpy as np
from torchsummary import summary
import random
from utils.util import merge_dataset, split_data_label, split_data_label_tv, down_sampling, WriteExcel, \
    augument_dataset, \
    make_one_shot_dataset, change_windowsize
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_CUDA_CACHE_PATH'] = '/home/cuda_cache'

torch.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument("--data_dir", default='/home/test_each_baseline/LIMU-BERT-Public-master-920/dataset')
parser.add_argument("--label_dir", default='/home/test_each_baseline/LIMU-BERT-Public-master-920/dataset')
parser.add_argument("-d", "--dataset", default='ourdata_20240328', type=str)
parser.add_argument("-td", "--target_dataset", default='uci', type=str)
parser.add_argument("-df", "--data_file", default='data_20_120_5activity.npy', type=str)
parser.add_argument("-lf", "--label_file", default='label_20_120_4activity.npy', type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--L", default=120, type=int)  # seq_len or window size
parser.add_argument("--N", default=6, type=int)  # num_channel
parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--model_path", default='/home/test_each_baseline/baseline/save_model')
parser.add_argument("--save_excel_path", default='/home/test_each_baseline/baseline/excel_result')
parser.add_argument("--mode", default='convert_pt_to_pt',
                    choices=['train', 'test', 'cross_dataset', 'convert_pt_to_pt'])
parser.add_argument("--method", default='TimeSeriesTransformer',
                    choices=['TimeSeriesTransformer'])
parser.add_argument("--train_rate", default=0.4)
parser.add_argument("--valid_rate", default=0.3)
parser.add_argument("--test_rate", default=0.3)
parser.add_argument("--original_sampling_rate", default=20)
parser.add_argument('-asr', "--aim_sampling_rate", default=20, choices=[20, 10, 5, 4, 2, 1], type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("-ps", '--patch_size', default=10, type=int)
parser.add_argument("-s", '--stride', default=10, type=int)
parser.add_argument('--is_mask', default=True, type=bool)
parser.add_argument('--is_one_shot', default=False, type=bool)
parser.add_argument('--aug_method', default="", type=str, choices=["", 'rotation_random'])
args = parser.parse_args()


class Main_Transformer:
    def __init__(self):
        print(args.mode)
        print('batch size:', str(args.batch_size))
        print('epoch:', str(args.epochs))

        self.model_path = args.model_path
        self.is_mask = args.is_mask

    def init_seed(self):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    def init_model(self):
        print('method:' + args.method)

        if args.mode == 'convert_pt_to_pt':
            is_trans_pt = True
        else:
            is_trans_pt = False

        model_dict = {
            'TimeSeriesTransformer': TimeSeriesTransformer(d_model=64, nhead=1, num_layers=1, dim_feedforward=32,
                                                           seq_length=args.L,
                                                           output_size=args.n_class, channel_size=args.N,
                                                           is_trans_pt=is_trans_pt, is_mask=self.is_mask,
                                                           filter_pos=1)
        }
        self.model = model_dict[args.method]

        if args.mode == 'convert_pt_to_pt':
            self.model = self.model.cpu()
        else:
            self.model = self.model.cuda()

    def init_dataset(self):

        if args.mode == 'train' or args.mode == 'test' or args.mode == 'convert_pt_to_pt':
            data = np.load(os.path.join(args.data_dir, args.dataset, args.data_file)).astype(np.float32)
            labels = np.load(os.path.join(args.data_dir, args.dataset, args.label_file)).astype(np.float32)
        elif args.mode == 'cross_dataset':
            # args.train_rate, args.valid_rate, args.test_rate = 0.0, 0.0, 1.0
            data = np.load(os.path.join(args.data_dir, args.target_dataset, args.data_file)).astype(np.float32)
            labels = np.load(os.path.join(args.data_dir, args.target_dataset, args.label_file)).astype(np.float32)

        data, labels = change_windowsize(data, labels, args.L)

        if args.aug_method != '':
            data, labels = augument_dataset(data, labels, args.aug_method)

        if args.mode != 'cross_dataset':
            data_train, label_train, data_vali, label_vali, data_test, label_test = split_data_label(data, labels,
                                                                                                     args.train_rate,
                                                                                                     args.valid_rate)
        else:
            data_train, label_train, data_vali, label_vali, data_test, label_test = split_data_label(data, labels, 0.0,
                                                                                                     0.0)

        if args.is_one_shot and args.mode == 'train':
            print('one_shot_traing_mode')
            data_train, label_train = make_one_shot_dataset(data_train, label_train)
            print('Train Size: %d, Vali Size: %d, Test Size: %d' % (
            label_train.shape[0], label_vali.shape[0], label_test.shape[0]))

        data_set_train = IMUDataset_fft(data_train, label_train, isNormalization=False)
        data_set_vali = IMUDataset_fft(data_vali, label_vali, isNormalization=False)
        data_set_test = IMUDataset_fft(data_test, label_test, isNormalization=False)

        if args.mode == 'train':
            self.train_loader = DataLoader(data_set_train, shuffle=True, batch_size=args.batch_size, drop_last=False)
            self.valid_loader = DataLoader(data_set_vali, shuffle=False, batch_size=args.batch_size, drop_last=False)
        self.test_loader = DataLoader(data_set_test, shuffle=False, batch_size=args.batch_size, drop_last=False)

        self.criteron = nn.CrossEntropyLoss().cuda()

    def train(self):
        summary(self.model, (args.N, args.L))

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        best_loss = 9999999.99
        print('epoch:', args.epochs)
        for epoch in range(args.epochs):
            self.model.train()
            for i, (x_orgin, x_fft, y) in enumerate(self.train_loader):
                x_orgin = x_orgin.cuda()
                x_fft = x_fft.cuda()
                y = y.cuda()
                output = self.model(x_orgin)
                loss = self.criteron(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = self.evaluate(self.valid_loader)
            print('epoch, loss:', epoch, loss)
            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(self.model.state_dict())
                os.makedirs('./save_model/', exist_ok=True)
                torch.save(best_model, self.model_path)

    def evaluate(self, val_iter):
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for i, (x_orgin, x_fft, y) in enumerate(val_iter):
                x_orgin = x_orgin.cuda()
                x_fft = x_fft.cuda()
                y = y.cuda()
                output = self.model(x_orgin)
                loss += self.criteron(output, y)
        return loss / len(val_iter)

    def test(self, test_iter):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            conf_matrix = torch.zeros(args.n_class, args.n_class)
            print(len(test_iter))
            for i, (x_orgin, x_fft, y) in enumerate(test_iter):
                x_orgin = x_orgin.cuda()
                x_fft = x_fft.cuda()
                y = y.cuda()
                output = self.model(x_orgin)
                predicted = torch.max(output, 1)[1]
                y_true = y_true + y.tolist()
                y_pred = y_pred + predicted.tolist()

        conf_matrix = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        return conf_matrix, acc, precision, recall, f1_macro, f1_micro

    def run(self):
        self.init_seed()
        self.init_dataset()
        self.init_model()
        if args.mode == 'train':
            self.model_path = os.path.join(args.model_path, args.dataset + '_' + args.method + '.pth')
            print('model will save at ' + self.model_path)
            self.train()
            self.model.load_state_dict(torch.load(self.model_path))
            conf_matrix, acc, precision, recall, f1_macro, f1_micro = self.test(self.test_loader)
            print('acc:{:.4f},precision:{:.4f},recall:{:.4f},f1_macor:{:.4f},f1_micor:{:.4f}'
                  .format(acc, precision, recall, f1_macro, f1_micro))
            torch.set_printoptions(sci_mode=False)
            print('conf_matrix')
            print(conf_matrix)
            return conf_matrix, acc, f1_macro
        elif args.mode == 'test':
            self.model_path = os.path.join(args.model_path, args.dataset + '_' + args.method + '.pth')
            self.model.load_state_dict(torch.load(self.model_path))
            conf_matrix, acc, precision, recall, f1_macro, f1_micro = self.test(self.test_loader)
            print('acc:', acc)
            print('confuse matrix:', conf_matrix)
        elif args.mode == 'cross_dataset':
            print('source_dataset:{},target_dataset:{}'.format(args.dataset, args.target_dataset))
            self.model_path = os.path.join(args.model_path, args.dataset + '_' + args.method + '.pth')
            print('model load from ' + self.model_path)
            self.model.load_state_dict(torch.load(self.model_path))
            conf_matrix, acc, precision, recall, f1_macro, f1_micro = self.test(self.test_loader)
            print('acc:{:.4f},precision:{:.4f},recall:{:.4f},f1_macor:{:.4f},f1_micor:{:.4f}'
                  .format(acc, precision, recall, f1_macro, f1_micro))
            torch.set_printoptions(sci_mode=False)
            print('conf_matrix')
            print(conf_matrix)
            return conf_matrix, acc, f1_macro
        elif args.mode == 'convert_pt_to_pt':
            # model = model_dict[args.method]
            # self.model = self.model.cpu()
            # model_name = 'ourdata_20240326_TimeSeriesTransformer.pth'
            model_name = f'{args.dataset}_{args.method}.pth'
            # self.model_path = os.path.join(args.model_path,args.dataset+'_'+args.method+'.pth')
            self.model_path = os.path.join(args.model_path, model_name)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            example_input = torch.rand(1, args.N, args.L)
            traced_script_module = torch.jit.trace(self.model, example_input)
            # traced_script_module.save(f'./target_pt_model/{args.dataset}_{args.method}.pt')
            # traced_script_module.save(f'./target_pt_model/ourdata_20240326_TimeSeriesTransformer.pt')
            traced_script_module.save(f'./target_pt_model/{args.dataset}_{args.method}.pt')
        return 0.0, 0.0  # return acc,f1

    def loop_train_and_test(self):
        we = WriteExcel('test.xlsx')

        dataset_list = ['uci', 'shoaib', 'motion', 'hhar']
        acc_sum_avg_list, f1_sum_avg_list = [], []

        for i in range(len(dataset_list)):
            args.mode = 'train'
            args.dataset = dataset_list[i]
            cm_temp, acc_temp, f1_temp = self.run()
            we.write_excel(args.dataset, args.dataset, str(cm_temp), acc_temp, f1_temp)

            temp_dataset_list = [item for item in dataset_list if item != args.dataset]
            args.mode = 'cross_dataset'
            acc_temp_avg_list, f1_temp_avg_list = [], []
            for target_dataset in temp_dataset_list:
                args.target_dataset = target_dataset
                cm_temp, acc_temp, f1_temp = self.run()
                we.write_excel(args.dataset, args.target_dataset, str(cm_temp), acc_temp, f1_temp)
                acc_temp_avg_list.append(acc_temp)
                f1_temp_avg_list.append(f1_temp)

            print('--------------------------------------------------------------')
            print(f'args.dataset:{args.dataset}')
            acc_temp_avg = sum(acc_temp_avg_list) / len(acc_temp_avg_list)
            f1_temp_avg = sum(f1_temp_avg_list) / len(f1_temp_avg_list)
            print('acc_avg:{:.4f},f1_avg:{:.4f}'.format(acc_temp_avg, f1_temp_avg))
            we.write_excel_avg(args.dataset, acc_temp_avg, f1_temp_avg)
            acc_sum_avg_list.append(acc_temp_avg)
            f1_sum_avg_list.append(f1_temp_avg)
            print('--------------------------------------------------------------')

        print('--------------------------------------------------------------')
        acc_sum_avg = sum(acc_sum_avg_list) / len(acc_sum_avg_list)
        f1_sum_avg = sum(f1_sum_avg_list) / len(f1_sum_avg_list)
        print('acc_sum_avg:{:.4f},f1_sum_avg:{:.4f}'.format(acc_sum_avg, f1_sum_avg))
        we.write_excel_avg('avg_all', acc_sum_avg, f1_sum_avg)
        print('--------------------------------------------------------------')
        we.save_excel(args.save_excel_path)


if __name__ == '__main__':
    mt = Main_Transformer()
    mt.run()






