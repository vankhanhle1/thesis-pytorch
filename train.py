from model import ABCNN
import torch
from torch import nn, optim
from helpers import weights_init, from_pickle_to_dataloader, load_ckp, save_ckp, metrics_sklearn
import numpy as np
import joblib
from livelossplot import PlotLosses

def train(train_data_path, val_data_path, dnn_model_path, svm_model_path, model_type, svm_fn, epoch_size, batch_size,
          d0, sentence_length,
          filter_width, conv_layers_count, conv_kernels_count, activation_fn, loss_fn, include_input_ap, l2_reg, lr,
          device):
    train_loader = from_pickle_to_dataloader(train_data_path, batch_size)
    val_loader = from_pickle_to_dataloader(val_data_path, batch_size)

    model = ABCNN(model_type, d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count, activation_fn,
                  include_input_ap).to(device)
    model.apply(weights_init)
    SVM = svm_fn

    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_reg)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_total_batch = len(train_loader)
    val_total_batch = len(val_loader)
    liveloss = PlotLosses()

    for epoch in range(epoch_size):
        print("epoch ", epoch + 1)
        logs = {}
        train_loss = 0.0
        svm_features_list = []
        labels_for_svm = []

        for data, labels in train_loader:
            x0 = data[:, 0, :, :].unsqueeze(1)
            x1 = data[:, 1, :, :].unsqueeze(1)

            x0 = x0.to(device)
            x1 = x1.to(device)

            labels = labels.to(device)
            predictions, features = model(x0, x1)
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

        epoch_train_loss_abcnn = float(train_loss / train_total_batch)
        print('epoch: ', epoch + 1, ' train_loss_abcnn: ', epoch_train_loss_abcnn)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, dnn_model_path)

        svm_features = np.concatenate(svm_features_list)
        svm_labels = np.concatenate(labels_for_svm)

        svm_labels = svm_labels.flatten()

        SVM.fit(svm_features, svm_labels)
        #         get accuracy and f1 for svm to monitor overfitting
        joblib.dump(SVM, svm_model_path)

        logs['train_loss_abcnn'] = epoch_train_loss_abcnn

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            val_f1 = 0.0

            for data, labels in val_loader:
                x0 = data[:, 0, :, :].unsqueeze(1)
                x1 = data[:, 1, :, :].unsqueeze(1)

                x0 = x0.to(device)
                111111111
                x1 = x1.to(device)
                labels = labels.to(device)
                predictions, features = model(x0, x1)
                loss = loss_fn(predictions.float(), labels)
                loss = loss.to(device)
                val_loss += loss.item()

                if device != "cpu":
                    features = features.cpu()
                    labels = labels.cpu()

                logits = SVM.predict(features.numpy())
                acc_metrics = metrics_sklearn(logits, labels.numpy().flatten())
                val_accuracy += acc_metrics[0]
                val_f1 += acc_metrics[1]

        epoch_val_loss_abcnn = float(val_loss / val_total_batch)
        epoch_val_accuracy_svm = float(val_accuracy / val_total_batch)
        epoch_val_f1_svm = float(val_f1 / val_total_batch)

        logs['val_loss_abcnn'] = epoch_val_loss_abcnn
        logs['val_acc'] = epoch_val_accuracy_svm
        logs['val_f1'] = epoch_val_f1_svm

        liveloss.update(logs)
        liveloss.send()


def continue_train(train_data_path, val_data_path, input_dnn_path, input_svm_path, output_dnn_path, output_svm_path,
                   model_type, epoch_size, batch_size, d0, sentence_length,
                   filter_width, conv_layers_count, conv_kernels_count, activation_fn, loss_fn, include_input_ap,
                   l2_reg, lr, device):
    train_loader = from_pickle_to_dataloader(train_data_path, batch_size)
    val_loader = from_pickle_to_dataloader(val_data_path, batch_size)

    model = ABCNN(model_type, d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count, activation_fn,
                  include_input_ap).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_reg)

    model, optimizer, start_epoch = load_ckp(input_dnn_path, model, optimizer)

    SVM = joblib.load(input_svm_path)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

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

        for data, labels in train_loader:
            x0 = data[:, 0, :, :].unsqueeze(1)
            x1 = data[:, 1, :, :].unsqueeze(1)

            x0 = x0.to(device)
            x1 = x1.to(device)

            labels = labels.to(device)
            predictions, features = model(x0, x1)
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

        epoch_train_loss_abcnn = float(train_loss / train_total_batch)
        checkpoint = {
            'epoch': real_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, output_dnn_path)

        svm_features = np.concatenate(svm_features_list)
        svm_labels = np.concatenate(labels_for_svm)

        svm_labels = svm_labels.flatten()

        SVM.fit(svm_features, svm_labels)
        joblib.dump(SVM, output_svm_path)

        logs['train_loss_abcnn'] = epoch_train_loss_abcnn

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            val_f1 = 0.0

            for data, labels in val_loader:
                x0 = data[:, 0, :, :].unsqueeze(1)
                x1 = data[:, 1, :, :].unsqueeze(1)

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

                logits = SVM.predict(features.numpy())
                acc_metrics = metrics_sklearn(logits, labels.numpy().flatten())
                val_accuracy += acc_metrics[0]
                val_f1 += acc_metrics[1]

        epoch_val_loss_abcnn = float(val_loss / val_total_batch)
        epoch_val_accuracy_svm = float(val_accuracy / val_total_batch)
        epoch_val_f1_svm = float(val_f1 / val_total_batch)

        logs['val_loss_abcnn'] = epoch_val_loss_abcnn
        logs['val_acc'] = epoch_val_accuracy_svm
        logs['val_f1'] = epoch_val_f1_svm

        liveloss.update(logs)
        liveloss.send()

