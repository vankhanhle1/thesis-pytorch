import json

import joblib
import numpy as np
import pandas as pd
import torch
from livelossplot import PlotLosses
from torch import optim
from torch.nn import BCELoss

from model import ABCNN
from network_helpers import weights_init
from train_helpers import from_pickle_to_dataloader, load_checkpoint, save_checkpoint, metrics_sklearn, predict, \
    EpochSum, update_epoch_metrics, load_checkpoint_for_test


def train_svm(train_data_path, val_data_path, model_type, svm_fn, epoch_size, batch_size,
              filter_width, conv_layers_count, conv_kernels_count, activation_fn, loss_fn,
              include_input_ap, include_additional_features, l2_reg, lr, decay_step_size, device, output_dnn_model_path,
              output_svm_model_path, output_metrics_path, resume_training, input_dnn_model_path='',
              input_svm_model_path='', input_metrics_path=''):
    train_loader, sentence_length, d0 = from_pickle_to_dataloader(train_data_path, batch_size, True)
    val_loader = from_pickle_to_dataloader(val_data_path, batch_size)

    model = ABCNN(model_type, d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count, activation_fn,
                  include_input_ap, True).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=0.5)

    if resume_training:
        model, optimizer, start_epoch = load_checkpoint(input_dnn_model_path, model, optimizer)
        SVM = joblib.load(input_svm_model_path)
        metrics_df = pd.read_csv(input_metrics_path)
    else:
        start_epoch = 0
        model.apply(weights_init)
        SVM = svm_fn
        metrics_df = pd.DataFrame(
            columns=['epoch', 'train_loss', 'train_accuracy', 'train_f1_0', 'train_f1_1', 'train_f1_weighted',
                     'val_loss', 'val_accuracy', 'val_f1_0', 'val_f1_1', 'val_f1_weighted'])

    train_total_batch = len(train_loader)
    val_total_batch = len(val_loader)
    liveloss = PlotLosses()

    for epoch in range(epoch_size):
        real_epoch = start_epoch + epoch + 1
        print("epoch ", real_epoch)
        logs = {}
        train_loss = 0.0
        svm_features_list = []
        labels_for_svm = []

        for X, Y in train_loader:
            x0 = X[:, 0, :, :].unsqueeze(1)
            x1 = X[:, 1, :, :].unsqueeze(1)

            additional_features = Y[:, 0:-1]
            labels = Y[:, -1].unsqueeze(1)

            x0 = x0.to(device)
            x1 = x1.to(device)
            labels = labels.to(device)

            predictions, features = model(x0, x1)
            if device == "cpu":
                features_for_svm = features.detach().numpy()
            else:
                features_for_svm = features.cpu().detach().numpy()
            if include_additional_features:
                features_for_svm = np.concatenate((features_for_svm, additional_features), axis=1)

            svm_features_list.append(features_for_svm)
            if device != "cpu":
                labels_for_svm.append(labels.cpu())
            else:
                labels_for_svm.append(labels)
            loss = loss_fn(predictions.float(), labels)
            loss = loss.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        epoch_train_loss = float(train_loss / train_total_batch)
        checkpoint = {
            'epoch': real_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint, output_dnn_model_path)

        svm_features = np.concatenate(svm_features_list)
        svm_labels = np.concatenate(labels_for_svm)

        svm_labels = svm_labels.flatten()

        SVM.fit(svm_features, svm_labels)
        train_logits = SVM.predict(svm_features)
        epoch_train_accuracy, epoch_train_f1_0, epoch_train_f1_1, epoch_train_f1_weighted = metrics_sklearn(
            train_logits, svm_labels)

        joblib.dump(SVM, output_svm_model_path)

        logs['loss'] = epoch_train_loss
        logs['accuracy'] = epoch_train_accuracy
        logs['f1_class_0'] = epoch_train_f1_0
        logs['f1_class_1'] = epoch_train_f1_1
        logs['f1_weighted'] = epoch_train_f1_weighted

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            val_f1_0 = 0.0
            val_f1_1 = 0.0
            val_f1_weighted = 0.0

            for X, Y in val_loader:
                x0 = X[:, 0, :, :].unsqueeze(1)
                x1 = X[:, 1, :, :].unsqueeze(1)

                additional_features = Y[:, 0:-1]
                labels = Y[:, -1].unsqueeze(1)

                x0 = x0.to(device)
                x1 = x1.to(device)
                labels = labels.to(device)

                predictions, features = model(x0, x1)
                loss = loss_fn(predictions.float(), labels)
                loss = loss.to(device)
                val_loss += loss.item()

                if device != "cpu":
                    features = features.cpu()
                    labels = labels.cpu()
                if include_additional_features:
                    features = np.concatenate((features, additional_features), axis=1)

                logits = SVM.predict(features)
                val_metrics = metrics_sklearn(logits, labels.numpy().flatten())
                val_accuracy += val_metrics[0]
                val_f1_0 += val_metrics[1]
                val_f1_1 += val_metrics[2]
                val_f1_weighted += val_metrics[3]

        epoch_val_loss = float(val_loss / val_total_batch)
        epoch_val_accuracy = float(val_accuracy / val_total_batch)
        epoch_val_f1_0 = float(val_f1_0 / val_total_batch)
        epoch_val_f1_1 = float(val_f1_1 / val_total_batch)
        epoch_val_f1_weighted = float(val_f1_weighted / val_total_batch)

        epoch_metrics = {'epoch': real_epoch,
                         'train_loss': epoch_train_loss,
                         'train_accuracy': epoch_train_accuracy,
                         'train_f1_0': epoch_train_f1_0,
                         'train_f1_1': epoch_train_f1_1,
                         'train_f1_weighted': epoch_train_f1_weighted,
                         'val_loss': epoch_val_loss,
                         'val_accuracy': epoch_val_accuracy,
                         'val_f1_0': epoch_val_f1_0,
                         'val_f1_1': epoch_val_f1_1,
                         'val_f1_weighted': epoch_val_f1_weighted}
        metrics_df = metrics_df.append(epoch_metrics, ignore_index=True)
        metrics_df.to_csv(output_metrics_path, index=False)

        logs['val_loss'] = epoch_val_loss
        logs['val_accuracy'] = epoch_val_accuracy
        logs['val_f1_class_0'] = epoch_val_f1_0
        logs['val_f1_class_1'] = epoch_val_f1_1
        logs['val_f1_weighted'] = epoch_val_f1_weighted

        liveloss.update(logs)
        liveloss.send()


def train_lr(train_data_path, val_data_path, model_type, epoch_size, batch_size,
             filter_width, conv_layers_count, conv_kernels_count,
             include_input_ap, include_additional_features, threshold, l2_reg, lr, decay_step_size, gamma, device,
             parameters_path,
             output_dnn_model_path, output_metrics_path, resume_training, additional_features_count=16,
             input_dnn_model_path='', input_metrics_path=''):
    parameters = dict(locals().items())
    with open(parameters_path, 'w') as f:
        f.write(json.dumps(parameters))

    torch.backends.cudnn.benchmark = True
    train_loader, sentence_length, d0 = from_pickle_to_dataloader(train_data_path, batch_size, True)
    fc_input_count = conv_layers_count * conv_kernels_count
    if include_additional_features:
        fc_input_count += additional_features_count
    model = ABCNN(model_type, d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count, fc_input_count,
                  include_input_ap, False).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=gamma)
    loss_fn = BCELoss()

    if resume_training:
        model, optimizer, start_epoch = load_checkpoint(input_dnn_model_path, model, optimizer)
        metrics_df = pd.read_csv(input_metrics_path)
    else:
        start_epoch = 0
        model.apply(weights_init)
        if val_data_path != 'null':
            metrics_df = pd.DataFrame(
                columns=['epoch', 'train_loss', 'train_accuracy', 'train_f1_weighted',
                         'train_f1_0', 'train_f1_1', 'train_precision_0',
                         'train_precision_1', 'train_recall_0', 'train_recall_1',
                         'val_loss', 'val_accuracy', 'val_f1_weighted',
                         'val_f1_0', 'val_f1_1', 'val_precision_0',
                         'val_precision_1', 'val_recall_0', 'val_recall_1'])
        else:
            metrics_df = pd.DataFrame(
                columns=['epoch', 'train_loss', 'train_accuracy', 'train_f1_weighted',
                         'train_f1_0', 'train_f1_1', 'train_precision_0',
                         'train_precision_1', 'train_recall_0', 'train_recall_1'])

    train_total_batch = len(train_loader)
    liveloss = PlotLosses()

    for epoch in range(epoch_size):
        real_epoch = start_epoch + epoch + 1
        print("epoch ", real_epoch)
        epoch_metrics = {'epoch': real_epoch}
        logs = {}
        train_epoch_sum = EpochSum()

        for X, Y in train_loader:
            predict(X, Y, device, include_additional_features, model, loss_fn, threshold, train_epoch_sum, 'train',
                    optimizer)

        scheduler.step()

        checkpoint = {
            'epoch': real_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint, output_dnn_model_path)
        if (real_epoch % 25 == 0):
            save_checkpoint(checkpoint, output_dnn_model_path + '_' + str(real_epoch))

        update_epoch_metrics(train_epoch_sum, train_total_batch, logs, epoch_metrics)

        if val_data_path != 'null':
            val_loader = from_pickle_to_dataloader(val_data_path, batch_size)
            val_total_batch = len(val_loader)

            model.eval()
            with torch.no_grad():
                val_epoch_sum = EpochSum()

                for X, Y in val_loader:
                    predict(X, Y, device, include_additional_features, model, loss_fn, threshold, val_epoch_sum, 'val',
                            optimizer)

            update_epoch_metrics(val_epoch_sum, val_total_batch, logs,
                                 epoch_metrics, 'val_')

        metrics_df = metrics_df.append(epoch_metrics, ignore_index=True)
        metrics_df.to_csv(output_metrics_path, index=False)

        liveloss.update(logs)
        liveloss.send()


def resume_train(train_data_path, val_data_path, epoch_size, parameters_path, output_dnn_model_path,
                 output_metrics_path, input_dnn_model_path, input_metrics_path):
    with open(parameters_path, 'r') as f:
        params = json.load(f)

    batch_size = params.get('batch_size')
    model_type = params.get('model_type')
    filter_width = params.get('filter_width')
    conv_layers_count = params.get('conv_layers_count')
    conv_kernels_count = params.get('conv_kernels_count')
    include_input_ap = params.get('include_input_ap')
    include_additional_features = params.get('include_additional_features')
    threshold = params.get('threshold')
    l2_reg = params.get('l2_reg')
    lr = params.get('lr')
    decay_step_size = params.get('decay_step_size')
    gamma = params.get('gamma')
    device = params.get('device')
    additional_features_count = params.get('additional_features_count')

    train_lr(train_data_path, val_data_path, model_type, epoch_size, batch_size, filter_width, conv_layers_count,
             conv_kernels_count, include_input_ap, include_additional_features, threshold, l2_reg, lr, decay_step_size,
             gamma, device, parameters_path, output_dnn_model_path, output_metrics_path, True,
             additional_features_count, input_dnn_model_path, input_metrics_path)


def inference(test_data_path, model_path, params_path, results_path, batch_size):
    with open(params_path, 'r') as f:
        params = json.load(f)

    test_loader, sentence_length, d0 = from_pickle_to_dataloader(test_data_path, batch_size, True)
    test_total_batch = len(test_loader)
    model_type = params.get('model_type')
    filter_width = params.get('filter_width')
    conv_layers_count = params.get('conv_layers_count')
    conv_kernels_count = params.get('conv_kernels_count')
    include_input_ap = params.get('include_input_ap')
    include_additional_features = params.get('include_additional_features')
    threshold = params.get('threshold')
    device = params.get('device')
    additional_features_count = params.get('additional_features_count')

    fc_input_count = conv_layers_count * conv_kernels_count
    if include_additional_features:
        fc_input_count += additional_features_count
    model = ABCNN(model_type, d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count,
                  fc_input_count, include_input_ap, False).to(device)
    model = load_checkpoint_for_test(model_path, model)
    loss_fn = BCELoss()

    metrics = {}
    logs = {}

    model.eval()
    with torch.no_grad():
        test_sum = EpochSum()

        for X, Y in test_loader:
            predict(X, Y, device, include_additional_features, model, loss_fn, threshold, test_sum, 'test', 0)
    update_epoch_metrics(test_sum, test_total_batch, logs, metrics, 'test_')
    with open(results_path, 'w') as file:
        file.write(json.dumps(metrics))
    print(metrics)
