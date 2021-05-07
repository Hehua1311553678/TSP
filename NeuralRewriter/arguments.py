# -*- coding: utf-8 -*-
# @Time    : 2021/4/23 9:35
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : arguments.py
# @Software: PyCharm
import argparse

def get_arg_parser(tsp_nodes_num = 20):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--model_dir', type=str, default='./checkpoints/model_0')
    parser.add_argument('--model_dir', type=str, default='D:/projects/TSP/NeuralRewriter/checkpoints/tsp_{}/model_0'.format(tsp_nodes_num))
    # parser.add_argument('--model_dir', type=str, default='./checkpoints/tsp_{}/model_0'.format(tsp_nodes_num))
    parser.add_argument('--input_format', type=str, default='DAG', choices=['seq', 'DAG'])
    parser.add_argument('--max_eval_size', type=int, default=1000)
    parser.add_argument('--load_model', type=str, default=None)
    # parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--resume', type=int, default=1200)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--train_proportion', type=float, default=1.0)

    parser.add_argument('--LSTM_hidden_size', type=int, default=512)
    parser.add_argument('--MLP_hidden_size', type=int, default=256)
    parser.add_argument('--param_init', type=float, default=0.1)
    parser.add_argument('--num_LSTM_layers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_sample_rewrite_pos', type=int, default=10)
    parser.add_argument('--num_sample_rewrite_op', type=int, default=10)
    parser.add_argument('--max_reduce_steps', type=int, default=50)
    parser.add_argument('--cont_prob', type=float, default=0.5)

    parser.add_argument('--keep_last_n', type=int, default=None)
    parser.add_argument('--eval_every_n', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    # parser.add_argument('--log_name', type=str, default='model_0.csv')
    parser.add_argument('--log_name', type=str, default='tsp_{}/model_0.csv'.format(tsp_nodes_num))

    data_group =  parser.add_argument_group('data')
    data_group.add_argument('--train_dataset', type=str, default='./data/tsp_{}/tsp_{}_train.json'.format(tsp_nodes_num, tsp_nodes_num))
    data_group.add_argument('--val_dataset', type=str, default='./data/tsp_{}/tsp_{}_val.json'.format(tsp_nodes_num, tsp_nodes_num))
    data_group.add_argument('--test_dataset', type=str, default='./data/tsp_{}/tsp_{}_test.json'.format(tsp_nodes_num, tsp_nodes_num))
    data_group.add_argument('--lr', type=float, default=5e-5)
    data_group.add_argument('--value_loss_coef', type=float, default=0.01)
    data_group.add_argument('--gamma', type=float, default=0.9)
    data_group.add_argument('--batch_size', type=int, default=64)
    data_group.add_argument('--num_MLP_layers', type=int, default=2)
    data_group.add_argument('--embedding_size', type=int, default=7)
    data_group.add_argument('--attention_size', type=int, default=16)

    output_trace_group = parser.add_argument_group('output_trace_option')
    output_trace_group.add_argument('--output_trace_flag', type=str, default='nop',
                                    choices=['succeed', 'fail', 'complete', 'nop'])
    output_trace_group.add_argument('--output_trace_option', type=str, default='both', choices=['pred', 'both'])
    output_trace_group.add_argument('--output_trace_file', type=str, default=None)

    train_group = parser.add_argument_group('train')
    train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
    train_group.add_argument('--lr_decay_steps', type=int, default=500)
    train_group.add_argument('--lr_decay_rate', type=float, default=0.9)
    train_group.add_argument('--gradient_clip', type=float, default=5.0)
    # train_group.add_argument('--num_epochs', type=int, default=10)
    train_group.add_argument('--num_epochs', type=int, default=1000)
    train_group.add_argument('--dropout_rate', type=float, default=0.0)

    return parser