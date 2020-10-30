import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import pandas as pd
from tensorboardX import SummaryWriter
from adabound import AdaBound

from helpers.utils import csv_to_x_y, MeshImageDataset, bikecsv_to_x_y, BikeImageDataset
from models.model import MultiLabelClassifer, MultilabelEmbeddingNet, Resent18EmbeddingNet, Resent50EmbeddingNet, \
    Densenet121EmbeddingNet
from models.trainer import fit_multilabel, extract_multilabel_embeddings, extract_torch_models_embeddings

RANDOM_STATE = 666
CUDA = torch.cuda.is_available()


def exp_baseline_tb(args):
    """

    :param args:
    :return:

    The baseline model is trained using the simple cnn, on 245m by 245m images
    """
    print(args)
    print('Training baseline multilabel Model with images on {}'.format(args.data_file))

    batch_size = args.batch_size
    n_epochs = args.epoch
    log_interval = args.log_interval
    lr = args.learning_rate
    embedding_size = args.embedding_size
    data_file = args.data_file
    optimizer_name = args.optimizer
    momentum = args.momentum

    writer_train = SummaryWriter(comment='multilabel_baseline_training')
    writer_test = SummaryWriter(comment='multilabel_baseline_training')

    loc_path = os.path.join(DATA_FOLDER_PATH, data_file)
    x, y = csv_to_x_y(pd.read_csv(loc_path))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=RANDOM_STATE)
    num_class = len(y_train[0])

    train_dataset = MeshImageDataset(x_train, y_train, IMAGE_FOLDER_PATH)
    val_dataset = MeshImageDataset(x_test, y_test, IMAGE_FOLDER_PATH)

    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             **kwargs)

    embedding_net = MultilabelEmbeddingNet(embedding_size=embedding_size)
    model = MultiLabelClassifer(embedding_net, num_class, embedding_size)

    if CUDA:
        model.cuda()

    loss_fn = nn.MultiLabelSoftMarginLoss()

    # choose the optimizer to train
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    # train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda,
    #                    log_interval, embedding_size, writer, start_epoch=0
    _, _, test_p, test_a, test_r, test_f1, test_a2, test_m = fit_multilabel(train_loader,
                                                                            val_loader, model,
                                                                            loss_fn,
                                                                            optimizer, scheduler,
                                                                            n_epochs, CUDA,
                                                                            log_interval,
                                                                            writer_train, writer_test)

    # print("Best precision = {} at epoch {};" \
    #       "Best accuracy = {} at epoch {};" \
    #       "Best recall = {} at epoch {};" \
    #       "Best f1 score = {} at epoch {}; ".format(max(test_p), test_p.index(max(test_p)),
    #                                                 max(test_a), test_p.index(max(test_a)),
    #                                                 max(test_r), test_p.index(max(test_r)),
    #                                                 max(test_f1), test_p.index(max(test_f1))))

    folder_path = os.path.join(MODEL_PATH, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + 'b_' + str(batch_size) +
                               '_eb_' + str(embedding_size) + '_epoch_' + str(n_epochs) + '_multilabel_baseline')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(model.state_dict(), os.path.join(folder_path, 'trained_model'))
    writer_train.close()
    writer_test.close()


def main(args):
    """

    :param args:
    :return:

    """
    print('Training pretrained {} with images on {}'.format(args.model_name, args.data_file))

    batch_size = args.batch_size
    n_epochs = args.epoch
    log_interval = args.log_interval
    lr = args.learning_rate
    data_file = args.data_file
    optimizer_name = args.optimizer
    momentum = args.momentum
    embedding_size = args.embedding_size

    model_name = args.model_name

    print('Train Parameters: \n'
          'Batch_size: {}; \n'
          'Epoches: {}; \n'
          'log_interval: {}; \n'
          'Learning Rate: {}: \n'
          'Data File: {}: \n'
          'Embedding Size: {}\n'
          'Model: {}'.format(batch_size, n_epochs, log_interval, lr, data_file, embedding_size, model_name))

    writer_train = SummaryWriter(comment='multilabel_training pretrain-{}-train_{}-{}'.format(model_name,
                                                                                              embedding_size,
                                                                                              data_file))
    writer_test = SummaryWriter(comment='multilabel_training pretrain-{}-test_{}-{}'.format(model_name,
                                                                                            embedding_size,
                                                                                            data_file))

    # Prepare the dataloader
    loc_path = os.path.join(DATA_FOLDER_PATH, data_file)
    x, y = csv_to_x_y(pd.read_csv(loc_path))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=RANDOM_STATE)

    num_class = len(y_train[0])
    train_dataset = MeshImageDataset(x_train, y_train, IMAGE_FOLDER_PATH, normalize=True)
    val_dataset = MeshImageDataset(x_test, y_test, IMAGE_FOLDER_PATH, normalize=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             **kwargs)

    # Prepare the model
    if model_name == 'Resnet-18':
        embedding_net = Resent18EmbeddingNet(embedding_size=embedding_size, pretrained=True)
    elif model_name == 'Resnet-50':
        embedding_net = Resent50EmbeddingNet(embedding_size=embedding_size, pretrained=True)
    elif model_name == 'Dense-121':
        embedding_net = Densenet121EmbeddingNet(embedding_size=embedding_size, pretrained=True)

    model = MultiLabelClassifer(embedding_net, num_class, embedding_size=embedding_size)

    if CUDA:
        model.cuda()

    loss_fn = nn.MultiLabelSoftMarginLoss()

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'AdaBound':
        optimizer = AdaBound(model.parameters(), lr=lr, betas=(0.9, 0.999),
                             final_lr=0.1, gamma=0.001, weight_decay=5e-4)
    elif optimizer_name == 'AMSBound':
        optimizer = AdaBound(model.parameters(), lr=lr, etas=(0.9, 0.999),
                             final_lr=0.1, gamma=0.001, weight_decay=5e-4, amsbound=True)

    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    # train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda,
    #                    log_interval, embedding_size, writer, start_epoch=0
    _, _, test_p, test_a, test_r, test_f1, test_a2, test_m = fit_multilabel(train_loader,
                                                                            val_loader, model,
                                                                            loss_fn,
                                                                            optimizer, scheduler,
                                                                            n_epochs, CUDA,
                                                                            log_interval,
                                                                            writer_train, writer_test)

    # print("Best precision = {} at epoch {};" \
    #       "Best accuracy = {} at epoch {};" \
    #       "Best recall = {} at epoch {};" \
    #       "Best f1 score = {} at epoch {}; ".format(max(test_p), test_p.index(max(test_p)),
    #                                                 max(test_a), test_p.index(max(test_a)),
    #                                                 max(test_r), test_p.index(max(test_r)),
    #                                                 max(test_f1), test_p.index(max(test_f1))))

    folder_path = os.path.join(MODEL_PATH, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + 'b_' + str(batch_size) +
                               '_eb_' + str(embedding_size) + '_epoch_' + str(n_epochs) + '_' + optimizer_name +
                               '_multilabel_pretrained_{}_'.format(model_name) + data_file[:-4])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(model.state_dict(), os.path.join(folder_path, 'trained_model'))
    writer_train.close()
    writer_test.close()


def exp_extract_torch_models_embeddings(args, bike=False):
    batch_size = args.batch_size
    data_file = args.data_file
    embedding_size = args.embedding_size
    trained_model_path = args.trained_model

    resnet = args.resnet

    print(args)

    loc_path = os.path.join(DATA_FOLDER_PATH, data_file)

    if bike:
        batch_size = 1
        x, y = bikecsv_to_x_y(pd.read_csv(loc_path))
        val_dataset = BikeImageDataset(x, y, IMAGE_FOLDER_PATH, normalize=True)
        name = 'bike'
    else:
        x, y = csv_to_x_y(pd.read_csv(loc_path))
        val_dataset = MeshImageDataset(x, y, IMAGE_FOLDER_PATH, normalize=True)
        name = 'mesh'

    num_class = len(y[0])
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             **kwargs)

    if resnet == '18':
        embedding_net = Resent18EmbeddingNet(embedding_size=embedding_size, pretrained=True)
    elif resnet == '50':
        embedding_net = Resent50EmbeddingNet(embedding_size=embedding_size, pretrained=True)
    model = MultiLabelClassifer(embedding_net, num_class, embedding_size=embedding_size)

    if CUDA:
        model.cuda()

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, trained_model_path, 'trained_model')))

    embeddings, meshcodes = extract_torch_models_embeddings(val_loader, model, CUDA, embedding_size)

    folder_path = os.path.join(MODEL_PATH, trained_model_path + '_embeddings_' + name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(os.path.join(folder_path, 'loc_vectors'), embeddings)
    with open(os.path.join(folder_path, 'meshcodes.txt'), 'w') as file:
        for mesh in meshcodes:
            file.write("{}\n".format(mesh))


def exp_extract_baseline_embeddings(args):
    print('Extracting embeddings from baseline multilabel Model with images on {}'.format(args.data_file))

    batch_size = 1
    embedding_size = args.embedding_size
    data_file = args.data_file
    trained_model_path = args.trained_model

    loc_path = os.path.join(DATA_FOLDER_PATH, data_file)
    x, y = bikecsv_to_x_y(pd.read_csv(loc_path))

    num_class = len(y[0])

    # val_dataset = MeshImageDataset(x, y, IMAGE_FOLDER_PATH)
    val_dataset = BikeImageDataset(x, y, IMAGE_FOLDER_PATH)
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             **kwargs)

    embedding_net = MultilabelEmbeddingNet(embedding_size=embedding_size)
    model = MultiLabelClassifer(embedding_net, num_class, embedding_size)

    if CUDA:
        model.cuda()

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, trained_model_path, 'trained_model')))
    embeddings, meshcodes = extract_multilabel_embeddings(val_loader, model, embedding_size, CUDA)

    folder_path = os.path.join(MODEL_PATH, trained_model_path + '_bike_embeddings_3')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(os.path.join(folder_path, 'loc_vectors'), embeddings)
    with open(os.path.join(folder_path, 'meshcodes.txt'), 'w') as file:
        for mesh in meshcodes:
            file.write("{}\n".format(mesh))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--data_file', default='example.csv', type=str,
                        help='The csv file name for locations to train')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size per epoch')
    parser.add_argument('-e', '--epoch', default=20, type=int, help='number of epochs to train')
    parser.add_argument('-log', '--log_interval', default=20, type=int,
                        help='the interval to print messages in a epoch')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='the learning rate')
    parser.add_argument('-es', '--embedding_size', default=8, type=int, help='the size of embedding from the model')
    parser.add_argument('--poi_set', type=str, help='specify the set of poi used to train')
    parser.add_argument('--trained_model', type=str, help='specify the set of poi used to train')
    parser.add_argument('--model_name', default='Resnet-18', type=str, help='specify which model will be used')

    # Optimizer args
    parser.add_argument('--optimizer', default='Adam', type=str, help='specify the optimizer to train')
    parser.add_argument('--momentum', default=0.9, type=float, help='specify the optimizer to train')

    # change the path to the map image folders
    DATA_FOLDER_PATH = './data/example'
	IMAGE_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, 'images')

	# change the path to where the trained model is saved
	MODEL_PATH = './trained_model'

    args = parser.parse_args()

    main(args)
