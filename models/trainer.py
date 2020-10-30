from torch.autograd import Variable
import numpy as np
import torch
from sklearn.metrics import average_precision_score
import os
import torch.nn as nn
from tqdm import tqdm
import time
import datetime
from helpers.metrics import accuracy_precision_recall_f1, mcc


def train_multilabel_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, writer, epoch):
    model.train()
    losses = []
    total_loss = 0

    y_true = []
    y_pred = []
    t0 = time.time()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        data = tuple(Variable(d) for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        # save the ground truth and predictions
        sigmoid_outputs = torch.sigmoid(*outputs)  # b, num_classes
        y_true.append(target.data.cpu().numpy())
        y_pred.append(sigmoid_outputs.data.cpu().numpy())



        loss_inputs = outputs

        if target is not None:
            target = Variable(target)
            target = (target,)
            loss_inputs += target

        # loss inputs and target  should all be b x num_class
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))

            # visual_embeddings, visual_labels, _ = extract_embeddings(visual_loader, model, embedding_size, cuda,
            #                                                          poi_len, label_encoded=True)
            # # we need 3 dimension for tensor to visualize it!
            n_it = (epoch * len(train_loader)) + batch_idx
            # writer.add_embedding(
            #     torch.from_numpy(visual_embeddings).float(),
            #     metadata=torch.from_numpy(visual_labels),
            #     global_step=n_it)
            writer.add_scalar('loss', loss.item(), n_it)
            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    trained_time = str(datetime.timedelta(seconds=round((time.time() - t0), 0)))
    # compute average_precision_score for validation dataset
    y_pred = np.where(np.concatenate(y_pred, axis=0) >= 0.5, 1.0, 0.0)
    y_true = np.asarray(np.concatenate(y_true, axis=0))
    a, p, r, f1, a2 = accuracy_precision_recall_f1(y_true, y_pred)
    m, _ = mcc(y_true, y_pred, 1)

    writer.add_scalar('precision', p, epoch)
    writer.add_scalar('accuracy', a, epoch)
    writer.add_scalar('recall', r, epoch)
    writer.add_scalar('f1', f1, epoch)
    writer.add_scalar('binary_accuracy', a2, epoch)
    writer.add_scalar('mcc', m, epoch)
    return total_loss, a, p, r, f1, a2, m, trained_time


def test_multilabel_epoch(test_loader, model, loss_fn, cuda, writer, epoch):
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        t1 = time.time()
        for batch_idx, (data, target, _) in enumerate(test_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            # save the ground truth and predictions
            sigmoid_outputs = torch.sigmoid(*outputs)  # b, num_classes
            y_true.append(target.data.cpu().numpy())
            y_pred.append(sigmoid_outputs.data.cpu().numpy())

            loss_inputs = outputs
            if target is not None:
                target = Variable(target)
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()
            n_it = (epoch * len(test_loader)) + batch_idx
            writer.add_scalar('loss', loss.item(), n_it)

    test_time = str(datetime.timedelta(seconds=round((time.time() - t1), 0)))

    # compute average_precision_score for validation dataset
    y_pred = np.where(np.concatenate(y_pred, axis=0) >= 0.5, 1.0, 0.0)
    y_true = np.asarray(np.concatenate(y_true, axis=0))
    a, p, r, f1, a2 = accuracy_precision_recall_f1(y_true, y_pred)
    m, _ = mcc(y_true, y_pred, 1)

    writer.add_scalar('precision', p, epoch)
    writer.add_scalar('accuracy', a, epoch)
    writer.add_scalar('recall', r, epoch)
    writer.add_scalar('f1', f1, epoch)
    writer.add_scalar('binary_accuracy', a2, epoch)
    writer.add_scalar('mcc', m, epoch)

    val_loss /= (batch_idx + 1)
    return val_loss, a, p, r, f1, a2, m, test_time


def fit_multilabel(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda,
                   log_interval, writer_train, writer_test, start_epoch=0, validate=True):
    for epoch in range(0, start_epoch):
        scheduler.step()

    train_losses = []
    val_losses = []
    test_p = []
    test_a = []
    test_r = []
    test_f1 = []
    test_a2 = []
    test_m = []
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, a, p, r, f1, a2, m, trained_time = train_multilabel_epoch(train_loader, model, loss_fn, optimizer,
                                                                              cuda,
                                                                              log_interval,
                                                                              writer_train,
                                                                              epoch)

        if validate:
            val_loss, val_a, val_p, val_r, val_f1, val_a2, val_m, test_time = test_multilabel_epoch(test_loader, model,
                                                                                                    loss_fn,
                                                                                                    cuda,
                                                                                                    writer_test,
                                                                                                    epoch)

            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}, precision: {:.4f}, accuracy: {:.4f}, ' \
                      'recall: {:.4f}, f1: {:.4f}, binary accuracy: {:.4f}, mcc: {:4f}; ' \
                      'Test set: Average loss: {:.4f}, ' \
                      'precision: {:.4f}, accuracy: {:.4f} ' \
                      'recall: {:.4f}, f1: {:.4f}, binary accuracy: {:.4f}, mcc: {:4f}; Train time: {}, ' \
                      'Test time: {}.'.format(epoch + 1,
                                              n_epochs,
                                              train_loss,
                                              p, a, r, f1, a2, m,
                                              val_loss,
                                              val_p, val_a, val_r, val_f1,
                                              val_a2, val_m,
                                              trained_time, test_time)

            print(message)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_p.append(val_p)
            test_a.append(val_a)
            test_r.append(val_r)
            test_f1.append(val_f1)
            test_a2.append(val_a2)
            test_m.append(val_m)
        else:
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}, precision: {:.4f}, accuracy: {:.4f}, ' \
                      'recall: {:.4f}, f1: {:.4f}, binary accuracy: {:.4f}, mcc: {:4f};Time: {}.'.format(epoch + 1,
                                                                                                         n_epochs,
                                                                                                         train_loss,
                                                                                                         p, a, r, f1,
                                                                                                         a2, m,
                                                                                                         trained_time)
            print(message)

    return train_losses, val_losses, test_p, test_a, test_r, test_f1, test_a2, test_m


def extract_multilabel_embeddings(dataloader, model, embedding_size, cuda):
    """
        Extract embeddings from self defined CNN models
    :param dataloader:
    :param model:
    :param embedding_size:
    :param cuda:
    :return:
    """
    model.eval()
    # 1D embedding, dataset_size by embedding_size
    embeddings = np.zeros((len(dataloader.dataset), embedding_size))
    meshcodes = []
    k = 0
    for images, _, meshcode in tqdm(dataloader):
        if cuda:
            images = images.cuda()
        embeddings[k:k + images.shape[0]] = model.get_embedding(images).data.cpu().numpy()
        k += images.shape[0]
        meshcodes += list(meshcode)

    return embeddings, meshcodes


def extract_torch_models_embeddings(dataloader, model, cuda, embedding_size=512):
    """
        Extract embeddings from pretrained/fine tuned resnet model from pytorch modules
    :return:
    """
    # model.eval()
    # embeddings = np.zeros((len(dataloader.dataset), embedding_size))
    #
    # one_embedding = torch.zeros(batch_size, embedding_size, 1, 1)
    #
    # def copy_data(m, i, o):
    #     one_embedding.copy_(o.data)
    #
    # layer = model._modules.get('avgpool')
    # h = layer.register_forward_hook(copy_data)
    #
    # meshcodes = []
    # k = 0
    # for images, _, meshcode in tqdm(dataloader):
    #     if cuda:
    #         images = images.cuda()
    #     _ = model(images)
    #     embeddings[k:k + images.shape[0]] = one_embedding.numpy()[:, :, 0, 0]  # batchsize x 512 x 1 x 1
    #     k += images.shape[0]
    #     meshcodes += list(meshcode)
    #
    # h.remove()
    # return embeddings, meshcodes

    model.eval()
    # 1D embedding, dataset_size by embedding_size
    embeddings = np.zeros((len(dataloader.dataset), embedding_size))
    labels = []
    k = 0
    for images, _, label in tqdm(dataloader):
        if cuda:
            images = images.cuda()
        embeddings[k:k + images.shape[0]] = model.get_embedding(images).data.cpu().numpy()
        k += images.shape[0]
        labels += list(label)

    return embeddings, labels
