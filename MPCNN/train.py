import json

import pandas as pd
import torch
from livelossplot import PlotLosses
from torch import optim
from torch.nn import BCELoss

from MPCNN.train_helpers import from_pickle_to_dataloader, load_checkpoint, save_checkpoint, predict, \
    EpochSum, update_epoch_metrics, load_checkpoint_for_test, weights_init
from model import MPCNN


def train_lr(train_data_path, val_data_path, epoch_size, save_interval, batch_size,
             n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units,
             dropout, ext_feats, attention, wide_conv,
             include_additional_features, optimizer_type, l2_reg, lr, epsilon, momentum,
             patience, decay_factor, device, threshold,
             parameters_path, output_dnn_model_path, output_metrics_path, resume_training=False,
             input_dnn_model_path='', input_metrics_path=''):
    parameters = dict(locals().items())
    with open(parameters_path, 'w') as f:
        f.write(json.dumps(parameters))

    train_loader, sentence_length, d0 = from_pickle_to_dataloader(train_data_path, batch_size, True)
    model = MPCNN(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units,
                  dropout, ext_feats, attention, wide_conv).to(device)
    optimizer = None
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg, eps=epsilon)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_reg)
    else:
        raise ValueError('optimizer not recognized: it should be either adam or sgd')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=decay_factor,
                                                           patience=patience)
    loss_fn = BCELoss()

    if resume_training:
        model, optimizer, start_epoch = load_checkpoint(input_dnn_model_path, model, optimizer)
        metrics_df = pd.read_csv(input_metrics_path)
    else:
        start_epoch = 0
        model.apply(weights_init)
        if val_data_path != 'null':
            metrics_df = pd.DataFrame(
                columns=['epoch', 'loss', 'accuracy', 'f1_weighted',
                         'f1_0', 'f1_1', 'precision_0',
                         'precision_1', 'recall_0', 'recall_1',
                         'val_loss', 'val_accuracy', 'val_f1_weighted',
                         'val_f1_0', 'val_f1_1', 'val_precision_0',
                         'val_precision_1', 'val_recall_0', 'val_recall_1'])
        else:
            metrics_df = pd.DataFrame(
                columns=['epoch', 'loss', 'accuracy', 'f1_weighted',
                         'f1_0', 'f1_1', 'precision_0',
                         'precision_1', 'recall_0', 'recall_1', ])

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

        checkpoint = {
            'epoch': real_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint, output_dnn_model_path)
        if (real_epoch % save_interval == 0):
            save_checkpoint(checkpoint, output_dnn_model_path + '_' + str(real_epoch))

        update_epoch_metrics(train_epoch_sum, train_total_batch, logs, epoch_metrics)
        scheduler.step(epoch_metrics['accuracy'])

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


def resume_train(train_data_path, val_data_path, epoch_size, save_interval, parameters_path, output_dnn_model_path,
                 output_metrics_path, input_dnn_model_path, input_metrics_path):
    with open(parameters_path, 'r') as f:
        params = json.load(f)

    batch_size = params.get('batch_size')
    optimizer_type = params.get('optimizer_type')
    l2_reg = params.get('l2_reg')
    epsilon = params.get('epsilon')
    momentum = params.get('momentum')
    patience = params.get('patience')
    lr = params.get('lr')
    decay_factor = params.get('decay_factor')
    n_word_dim = params.get('n_word_dim')
    n_holistic_filters = params.get('n_holistic_filters')
    n_per_dim_filters = params.get('n_per_dim_filters')
    filter_widths = params.get('filter_widths')
    hidden_layer_units = params.get('hidden_layer_units')
    dropout = params.get('dropout')
    ext_feats = params.get('ext_feats')
    attention = params.get('attention')
    wide_conv = params.get('wide_conv')
    include_additional_features = params.get('include_additional_features')
    device = params.get('device')
    threshold = params.get('threshold')

    train_lr(train_data_path, val_data_path, epoch_size, save_interval, batch_size,
             n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units,
             dropout, ext_feats, attention, wide_conv,
             include_additional_features, optimizer_type, l2_reg, lr, epsilon, momentum,
             patience, decay_factor, device, threshold,
             parameters_path, output_dnn_model_path, output_metrics_path, True,
             input_dnn_model_path, input_metrics_path)


def inference(test_data_path, model_path, params_path, results_path, batch_size):
    with open(params_path, 'r') as f:
        params = json.load(f)

    test_loader, sentence_length, d0 = from_pickle_to_dataloader(test_data_path, batch_size, True)
    test_total_batch = len(test_loader)
    n_word_dim = params.get('n_word_dim')
    n_holistic_filters = params.get('n_holistic_filters')
    n_per_dim_filters = params.get('n_per_dim_filters')
    filter_widths = params.get('filter_widths')
    hidden_layer_units = params.get('hidden_layer_units')
    dropout = params.get('dropout')
    ext_feats = params.get('ext_feats')
    attention = params.get('attention')
    wide_conv = params.get('wide_conv')
    include_additional_features = params.get('include_additional_features')
    device = params.get('device')
    threshold = params.get('threshold')

    model = MPCNN(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units,
                  dropout, ext_feats, attention, wide_conv).to(device)
    model = load_checkpoint_for_test(model_path, model)
    loss_fn = BCELoss()

    metrics = {}
    logs = {}

    model.eval()
    with torch.no_grad():
        test_sum = EpochSum()

        for X, Y in test_loader:
            predict(X, Y, device, include_additional_features, model, loss_fn, threshold, test_sum, 'test', None)
    update_epoch_metrics(test_sum, test_total_batch, logs, metrics, 'test_')
    with open(results_path, 'w') as file:
        file.write(json.dumps(metrics))
    print(metrics)
