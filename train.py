import joblib
import numpy as np
import pandas as pd
import torch
from livelossplot import PlotLosses
from torch import optim

from model import ABCNN
from network_helpers import weights_init
from train_helpers import from_pickle_to_dataloader, load_checkpoint, save_checkpoint, metrics_sklearn


def train(train_data_path, val_data_path, model_type, svm_fn, epoch_size, batch_size,
          d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count, activation_fn, loss_fn,
          include_input_ap, include_additional_features, l2_reg, lr, decay_step_size, device, output_dnn_model_path,
          output_svm_model_path, output_metrics_path, resume_training, input_dnn_model_path='',
          input_svm_model_path='', input_metrics_path=''):
    train_loader = from_pickle_to_dataloader(train_data_path, batch_size)
    val_loader = from_pickle_to_dataloader(val_data_path, batch_size)

    model = ABCNN(model_type, d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count, activation_fn,
                  include_input_ap).to(device)
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
        metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_accuracy', 'train_f1_0', 'train_f1_1',
                         'val_loss', 'val_accuracy', 'val_f1_0', 'val_f1_1'])

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
            if include_additional_features:
                features = torch.cat((features, additional_features), 1)
            if device == "cpu":
                features_for_svm = features.detach().numpy()
            else:
                features_for_svm = features.cpu().detach().numpy()

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
        print('predict', train_logits)
        print('truth', svm_labels)
        epoch_train_accuracy, epoch_train_f1_0, epoch_train_f1_1 = metrics_sklearn(train_logits, svm_labels)

        joblib.dump(SVM, output_svm_model_path)

        logs['loss'] = epoch_train_loss
        logs['accuracy'] = epoch_train_accuracy
        logs['f1_class_0'] = epoch_train_f1_0
        logs['f1_class_1'] = epoch_train_f1_1

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            val_f1_0 = 0.0
            val_f1_1 = 0.0

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

                if include_additional_features:
                    features = torch.cat((features, additional_features), 1)

                if device != "cpu":
                    features = features.cpu()
                    labels = labels.cpu()

                logits = SVM.predict(features.numpy())
                val_metrics = metrics_sklearn(logits, labels.numpy().flatten())
                val_accuracy += val_metrics[0]
                val_f1_0 += val_metrics[1]
                val_f1_1 += val_metrics[2]

        epoch_val_loss = float(val_loss / val_total_batch)
        epoch_val_accuracy = float(val_accuracy / val_total_batch)
        epoch_val_f1_0 = float(val_f1_0 / val_total_batch)
        epoch_val_f1_1 = float(val_f1_1 / val_total_batch)

        epoch_metrics = {'epoch': real_epoch,
                         'train_loss': epoch_train_loss,
                         'train_accuracy': epoch_train_accuracy,
                         'train_f1_0': epoch_train_f1_0,
                         'train_f1_1': epoch_train_f1_1,
                         'val_loss': epoch_val_loss,
                         'val_accuracy': epoch_val_accuracy,
                         'val_f1_0': epoch_val_f1_0,
                         'val_f1_1': epoch_val_f1_1}
        metrics_df = metrics_df.append(epoch_metrics, ignore_index=True)
        metrics_df.to_csv(output_metrics_path,index=False)

        logs['val_loss'] = epoch_val_loss
        logs['val_accuracy'] = epoch_val_accuracy
        logs['val_f1_class_0'] = epoch_val_f1_0
        logs['val_f1_class_1'] = epoch_val_f1_1

        liveloss.update(logs)
        liveloss.send()


#
# def continue_train(train_data_path, val_data_path, input_dnn_path, input_svm_path, output_dnn_path, output_svm_path,
#                    model_type, epoch_size, batch_size, d0, sentence_length, filter_width, conv_layers_count,
#                    conv_kernels_count, activation_fn, loss_fn, include_input_ap, include_additional_features,
#                    l2_reg, lr, decay_step_size, device):
#     train_loader = from_pickle_to_dataloader(train_data_path, batch_size)
#     val_loader = from_pickle_to_dataloader(val_data_path, batch_size)
#
#     model = ABCNN(model_type, d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count, activation_fn,
#                   include_input_ap).to(device)
#     optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_reg)
#
#     model, optimizer, start_epoch = load_ckp(input_dnn_path, model, optimizer)
#
#     SVM = joblib.load(input_svm_path)
#
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=0.5)
#
#     train_total_batch = len(train_loader)
#     val_total_batch = len(val_loader)
#     liveloss = PlotLosses()
#
#     for epoch in range(epoch_size):
#         real_epoch = start_epoch + epoch + 1
#         print("epoch ", real_epoch)
#         logs = {}
#         train_loss = 0.0
#         svm_features_list = []
#         labels_for_svm = []
#
#         for data, labels in train_loader:
#             x0 = data[:, 0, :, :].unsqueeze(1)
#             x1 = data[:, 1, :, :].unsqueeze(1)
#
#             x0 = x0.to(device)
#             x1 = x1.to(device)
#
#             labels = labels.to(device)
#             predictions, features = model(x0, x1)
#             if device == "cpu":
#                 features_for_svm = features.detach().numpy()
#             else:
#                 features_for_svm = features.cpu().detach().numpy()
#
#             svm_features_list.append(features_for_svm)
#             if device != "cpu":
#                 labels_for_svm.append(labels.cpu())
#             else:
#                 labels_for_svm.append(labels)
#             loss = loss_fn(predictions.float(), labels)
#             loss = loss.to(device)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#
#         scheduler.step()
#
#         epoch_train_loss_abcnn = float(train_loss / train_total_batch)
#         checkpoint = {
#             'epoch': real_epoch,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict()
#         }
#         save_ckp(checkpoint, output_dnn_path)
#
#         svm_features = np.concatenate(svm_features_list)
#         svm_labels = np.concatenate(labels_for_svm)
#
#         svm_labels = svm_labels.flatten()
#
#         SVM.fit(svm_features, svm_labels)
#         joblib.dump(SVM, output_svm_path)
#
#         logs['train_loss_abcnn'] = epoch_train_loss_abcnn
#
#         model.eval()
#         with torch.no_grad():
#             val_loss = 0.0
#             val_accuracy = 0.0
#             val_f1 = 0.0
#
#             for data, labels in val_loader:
#                 x0 = data[:, 0, :, :].unsqueeze(1)
#                 x1 = data[:, 1, :, :].unsqueeze(1)
#
#                 x0 = x0.to(device)
#                 x1 = x1.to(device)
#                 labels = labels.to(device)
#                 predictions, features = model(x0, x1)
#                 loss = loss_fn(predictions.float(), labels)
#                 loss = loss.to(device)
#                 val_loss += loss.item()
#
#                 if device != "cpu":
#                     features = features.cpu()
#                     labels = labels.cpu()
#
#                 logits = SVM.predict(features.numpy())
#                 acc_metrics = metrics_sklearn(logits, labels.numpy().flatten())
#                 val_accuracy += acc_metrics[0]
#                 val_f1 += acc_metrics[1]
#
#         epoch_val_loss_abcnn = float(val_loss / val_total_batch)
#         epoch_val_accuracy_svm = float(val_accuracy / val_total_batch)
#         epoch_val_f1_svm = float(val_f1 / val_total_batch)
#
#         logs['val_loss_abcnn'] = epoch_val_loss_abcnn
#         logs['val_acc'] = epoch_val_accuracy_svm
#         logs['val_f1'] = epoch_val_f1_svm
#
#         liveloss.update(logs)
#         liveloss.send()
