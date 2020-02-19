# Marcus Vinicius Sousa Leite de Carvalho
# marcus.decarvalho@ntu.edu.sg
# ivsucram@gmail.com
#
# NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License Agreement
# Non-Commercial Use Only
# This NTUITIVE License Agreement, including all exhibits ("NTUITIVE-LA") is a legal agreement between you and NTUITIVE (or “we”) located at 71 Nanyang Drive, NTU Innovation Centre, #01-109, Singapore 637722, a wholly owned subsidiary of Nanyang Technological University (“NTU”) for the software or data identified above, which may include source code, and any associated materials, text or speech files, associated media and "online" or electronic documentation and any updates we provide in our discretion (together, the "Software").
#
# By installing, copying, or otherwise using this Software, found at https://github.com/Ivsucram/ATL_Matlab, you agree to be bound by the terms of this NTUITIVE-LA.  If you do not agree, do not install copy or use the Software. The Software is protected by copyright and other intellectual property laws and is licensed, not sold.   If you wish to obtain a commercial royalty bearing license to this software please contact us at marcus.decarvalho@ntu.edu.sg.
#
# SCOPE OF RIGHTS:
# You may use, copy, reproduce, and distribute this Software for any non-commercial purpose, subject to the restrictions in this NTUITIVE-LA. Some purposes which can be non-commercial are teaching, academic research, public demonstrations and personal experimentation. You may also distribute this Software with books or other teaching materials, or publish the Software on websites, that are intended to teach the use of the Software for academic or other non-commercial purposes.
# You may not use or distribute this Software or any derivative works in any form for commercial purposes. Examples of commercial purposes would be running business operations, licensing, leasing, or selling the Software, distributing the Software for use with commercial products, using the Software in the creation or use of commercial products or any other activity which purpose is to procure a commercial gain to you or others.
# If the Software includes source code or data, you may create derivative works of such portions of the Software and distribute the modified Software for non-commercial purposes, as provided herein.
# If you distribute the Software or any derivative works of the Software, you will distribute them under the same terms and conditions as in this license, and you will not grant other rights to the Software or derivative works that are different from those provided by this NTUITIVE-LA.
# If you have created derivative works of the Software, and distribute such derivative works, you will cause the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.
#
# You may not distribute this Software or any derivative works.
# In return, we simply require that you agree:
# 1.	That you will not remove any copyright or other notices from the Software.
# 2.	That if any of the Software is in binary format, you will not attempt to modify such portions of the Software, or to reverse engineer or decompile them, except and only to the extent authorized by applicable law.
# 3.	That NTUITIVE is granted back, without any restrictions or limitations, a non-exclusive, perpetual, irrevocable, royalty-free, assignable and sub-licensable license, to reproduce, publicly perform or display, install, use, modify, post, distribute, make and have made, sell and transfer your modifications to and/or derivative works of the Software source code or data, for any purpose.
# 4.	That any feedback about the Software provided by you to us is voluntarily given, and NTUITIVE shall be free to use the feedback as it sees fit without obligation or restriction of any kind, even if the feedback is designated by you as confidential.
# 5.	THAT THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
# 6.	THAT NEITHER NTUITIVE NOR NTU NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS NTUITIVE-LA, INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
# 7.	That we have no duty of reasonable care or lack of negligence, and we are not obligated to (and will not) provide technical support for the Software.
# 8.	That if you breach this NTUITIVE-LA or if you sue anyone over patents that you think may apply to or read on the Software or anyone's use of the Software, this NTUITIVE-LA (and your license and rights obtained herein) terminate automatically.  Upon any such termination, you shall destroy all of your copies of the Software immediately.  Sections 3, 4, 5, 6, 7, 8, 11 and 12 of this NTUITIVE-LA shall survive any termination of this NTUITIVE-LA.
# 9.	That the patent rights, if any, granted to you in this NTUITIVE-LA only apply to the Software, not to any derivative works you make.
# 10.	That the Software may be subject to U.S. export jurisdiction at the time it is licensed to you, and it may be subject to additional export or import laws in other places.  You agree to comply with all such laws and regulations that may apply to the Software after delivery of the software to you.
# 11.	That all rights not expressly granted to you in this NTUITIVE-LA are reserved.
# 12.	That this NTUITIVE-LA shall be construed and controlled by the laws of the Republic of Singapore without regard to conflicts of law.  If any provision of this NTUITIVE-LA shall be deemed unenforceable or contrary to law, the rest of this NTUITIVE-LA shall remain in full effect and interpreted in an enforceable manner that most nearly captures the intent of the original language.
#
# Copyright (c) NTUITIVE. All rights reserved.

from DataManipulator import DataManipulator
from NeuralNetwork import NeuralNetwork
from AutoEncoder import DenoisingAutoEncoder
from AGMM import AGMM
from MySingletons import MyDevice, TorchDevice
from colorama import Fore, Back, Style
from itertools import cycle

import numpy as np
import matplotlib.pylab as plt

import math
import torch
import time


def copy_weights(source: NeuralNetwork, target: NeuralNetwork, layer_numbers=[1], copy_moment: bool = True):
    for layer_number in layer_numbers:
        layer_number -= 1
        if layer_number >= source.number_hidden_layers:
            target.output_weight = source.output_weight
            target.output_bias = source.output_bias
            if copy_moment:
                target.output_momentum = source.output_momentum
                target.output_bias_momentum = source.output_bias_momentum
        else:
            target.weight[layer_number] = source.weight[layer_number]
            target.bias[layer_number] = source.bias[layer_number]
            if copy_moment:
                target.momentum[layer_number] = source.momentum[layer_number]
                target.bias_momentum[layer_number] = source.bias_momentum[layer_number]


def grow_nodes(*networks):
    origin = networks[0]
    if origin.growable[origin.number_hidden_layers]:
        if origin.get_agmm() is None:
            nodes = 1
        else:
            nodes = origin.get_agmm().M()
        for i in range(nodes):
            for network in networks:
                network.grow_node(origin.number_hidden_layers)
        return True
    else:
        return False


def prune_nodes(*networks):
    origin = networks[0]
    if origin.prunable[origin.number_hidden_layers][0] >= 0:
        nodes_to_prune = origin.prunable[origin.number_hidden_layers].tolist()
        for network in networks:
            for node_to_prune in nodes_to_prune[::-1]:
                network.prune_node(origin.number_hidden_layers, node_to_prune)


def width_evolution(network: NeuralNetwork, x: torch.tensor, y: torch.tensor = None, agmm: AGMM = None, train_agmm: bool = False):
    if y is None:
        y = x

    if agmm is not None:
        network.set_agmm(agmm)
        if train_agmm:
            network.forward_pass(x)
            network.run_agmm(x, y)

    network.feedforward(x, y)
    network.width_adaptation_stepwise(y)


def discriminative(network: NeuralNetwork, x: torch.tensor, y: torch.tensor = None, agmm: AGMM = None):
    if agmm is not None:
        network.set_agmm(agmm)
    if y is None:
        y = x

    network.train(x, y)


def generative(network: NeuralNetwork, x: torch.tensor, y: torch.tensor = None, agmm: AGMM = None, is_tied_weight=False, noise_ratio=0.1, glw_epochs: int = 1):
    if agmm is not None:
        network.set_agmm(agmm)
    if y is None:
        y = x

    network.greedy_layer_wise_pretrain(x=x, number_epochs=glw_epochs, noise_ratio=0.0)
    network.train(x=x, y=y, noise_ratio=noise_ratio, is_tied_weight=is_tied_weight)


def test(network: NeuralNetwork, x: torch.tensor, y: torch.tensor = None, is_source: bool = False, is_discriminative: bool = False, metrics=None):
    with torch.no_grad():
        if y is None:
            y = x
        network.test(x=x, y=y)

        if is_source:
            if is_discriminative:
                metrics['classification_rate_source'].append(network.classification_rate)
                metrics['classification_source_loss'].append(float(network.loss_value))
            else:
                metrics['reconstruction_source_loss'].append(float(network.loss_value))
        else:
            if is_discriminative:
                metrics['classification_rate_target'].append(network.classification_rate)
                metrics['classification_target_loss'].append(float(network.loss_value))
            else:
                metrics['reconstruction_target_loss'].append(float(network.loss_value))


def force_same_size(a_tensor, b_tensor, shuffle=True, strategy='min'):
    common = np.min([a_tensor.shape[0], b_tensor.shape[0]])

    if shuffle:
        a_tensor = a_tensor[torch.randperm(a_tensor.shape[0])]
        b_tensor = b_tensor[torch.randperm(b_tensor.shape[0])]

    if strategy == 'max':
        if math.ceil(a_tensor.shape[0] / common) <= math.ceil(b_tensor.shape[0] / common):
            b_tensor = torch.stack(list(target for target, source in zip(b_tensor[torch.randperm(b_tensor.shape[0])], cycle(a_tensor[torch.randperm(a_tensor.shape[0])]))))
            a_tensor = torch.stack(list(source for target, source in zip(b_tensor[torch.randperm(b_tensor.shape[0])], cycle(a_tensor[torch.randperm(a_tensor.shape[0])]))))
        else:
            b_tensor = torch.stack(list(target for target, source in zip(cycle(b_tensor[torch.randperm(b_tensor.shape[0])]), a_tensor[torch.randperm(a_tensor.shape[0])])))
            a_tensor = torch.stack(list(source for target, source in zip(cycle(b_tensor[torch.randperm(b_tensor.shape[0])]), a_tensor[torch.randperm(a_tensor.shape[0])])))

    elif strategy == 'min':
        a_tensor = a_tensor[:common]
        b_tensor = b_tensor[:common]

    if shuffle:
        a_tensor = a_tensor[torch.randperm(a_tensor.shape[0])]
        b_tensor = b_tensor[torch.randperm(b_tensor.shape[0])]

    return a_tensor, b_tensor


def kl(ae: NeuralNetwork, x_source: torch.tensor, x_target: torch.tensor):
    x_source, x_target = force_same_size(x_source, x_target)

    ae.reset_grad()
    kl_loss = torch.nn.functional.kl_div(ae.forward_pass(x_target).layer_value[1],
                                         ae.forward_pass(x_source).layer_value[1])

    kl_loss.backward()
    ae.weight[0] = ae.weight[0] - ae.learning_rate * ae.weight[0].grad
    ae.bias[0] = ae.bias[0] - ae.learning_rate * ae.bias[0].grad

    return kl_loss.detach().cpu().numpy()


def print_annotation(lst):
    def custom_range(xx):
        return range(0, len(xx), int(len(xx) * 0.25) - 1)

    for idx in custom_range(lst):
        pos = lst[idx] if isinstance(lst[idx], (int, float)) else lst[idx][0]
        plt.annotate(format(pos, '.2f'), (idx, pos))
    pos = lst[-1] if isinstance(lst[-1], (int, float)) else lst[-1][0]
    plt.annotate(format(pos, '.2f'), (len(lst), pos))


def plot_time(train, test, annotation=True):
    plt.title('Processing time')
    plt.ylabel('Seconds')
    plt.xlabel('Minibatches')

    plt.plot(train, linewidth=1, label=('Train time Mean | Accumulative %f | %f' % (np.mean(train), np.sum(train))))
    plt.plot(test, linewidth=1, label=('Test time Mean | Accumulative %f | %f' % (np.mean(test), np.sum(test))))
    plt.legend()

    if annotation:
        print_annotation(train)
        print_annotation(test)

    plt.tight_layout()
    plt.show()


def plot_agmm(agmm_source, agmm_target, annotation=True):
    plt.title('AGMM evolution')
    plt.ylabel('GMMs')
    plt.xlabel('Samples')

    plt.plot(agmm_source, linewidth=1, label=('AGMM Source Discriminative Mean: %f' % (np.mean(agmm_source))))
    plt.plot(agmm_target, linewidth=1, label=('AGMM Target Generative Mean: %f' % (np.mean(agmm_target))))
    plt.legend()

    if annotation:
        print_annotation(agmm_source)
        print_annotation(agmm_target)

    plt.tight_layout()
    plt.show()


def plot_node_evolution(nodes, annotation=True):
    plt.title('Node evolution')
    plt.ylabel('Nodes')
    plt.xlabel('Minibatches')

    plt.plot(nodes, linewidth=1,
             label=('Hidden Layer Mean | Final: %f | %d' % (np.mean(nodes), nodes[-1])))
    plt.legend()

    if annotation:
        print_annotation(nodes)

    plt.tight_layout()
    plt.show()


def plot_losses(classification_source_loss, classification_target_loss, reconstruction_source_loss,
                reconstruction_target_loss, annotation=True):
    plt.title('Losses evolution')
    plt.ylabel('Loss value')
    plt.xlabel('Minibatches')

    plt.plot(classification_source_loss, linewidth=1,
             label=('Classification Source Loss mean: %f' % (np.mean(classification_source_loss))))
    plt.plot(classification_target_loss, linewidth=1,
             label=('Classification Target Loss mean: %f' % (np.mean(classification_target_loss))))
    plt.plot(reconstruction_source_loss, linewidth=1,
             label=('Reconstruction Source Loss mean: %f' % (np.mean(reconstruction_source_loss))))
    plt.plot(reconstruction_target_loss, linewidth=1,
             label=('Reconstruction Target Loss mean: %f' % (np.mean(reconstruction_target_loss))))
    plt.legend()

    if annotation:
        print_annotation(classification_source_loss)
        print_annotation(classification_target_loss)
        print_annotation(reconstruction_source_loss)
        print_annotation(reconstruction_target_loss)

    plt.tight_layout()
    plt.show()


def plot_classification_rates(source_rate, target_rate, annotation=True):
    plt.title('Source and Target Classification Rates')
    plt.ylabel('Classification Rate')
    plt.xlabel('Minibatches')

    plt.plot(source_rate, linewidth=1, label=('Source CR mean: %f' % (np.mean(source_rate))))
    plt.plot(target_rate, linewidth=1, label=('Target CR mean: %f' % (np.mean(target_rate))))

    if annotation:
        print_annotation(source_rate)
        print_annotation(target_rate)

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_ns(bias, var, ns, annotation=True):
    plt.plot(bias, linewidth=1, label=('Bias2 mean: %f' % (np.mean(bias))))
    plt.plot(var, linewidth=1, label=('Variance mean: %f' % (np.mean(var))))
    plt.plot(ns, linewidth=1, label=('Network Significance mean: %f' % (np.mean(ns))))
    plt.legend()

    if annotation:
        print_annotation(bias)
        print_annotation(var)
        print_annotation(ns)

    plt.tight_layout()
    plt.show()


def plot_discriminative_network_significance(bias, var, annotation=True):
    plt.title('Discriminative Source BIAS2, VAR, NS')
    plt.ylabel('Value')
    plt.xlabel('Sample')

    plot_ns(bias, var, (np.array(bias) + np.array(var)).tolist(), annotation)


def plot_generative_network_significance(bias, var, annotation=True, is_source=True):
    if is_source:
        plt.title('Generative Source BIAS2, VAR, NS')
    else:
        plt.title('Generative Target BIAS2, VAR, NS')
    plt.ylabel('Value')
    plt.xlabel('Sample')

    plot_ns(bias, var, (np.array(bias) + np.array(var)).tolist(), annotation)


def ATL(epochs: int = 1, n_batch: int = 1000, device='cpu'):
    def print_metrics(minibatch, metrics, nn, ae, Xs, Xt):
        print('Minibatch: %d | Execution time (dataset load/pre-processing + model run): %f' % (minibatch, time.time() - metrics['start_execution_time']))
        if minibatch > 1:
            string_max = '' + Fore.GREEN + 'Max' + Style.RESET_ALL
            string_mean = '' + Fore.YELLOW + 'Mean' + Style.RESET_ALL
            string_min = '' + Fore.RED + 'Min' + Style.RESET_ALL
            string_now = '' + Fore.BLUE + 'Now' + Style.RESET_ALL
            string_accu = '' + Fore.MAGENTA + 'Accu' + Style.RESET_ALL

            print(('Total of samples:' + Fore.BLUE + ' %d Source' + Style.RESET_ALL +' |' + Fore.RED +' %d Target' + Style.RESET_ALL) % (Xs.shape[0], Xt.shape[0]))
            print(('%s %s %s %s %s Training time:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Fore.MAGENTA + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now, string_accu,
                      np.max(metrics['train_time']),
                      np.mean(metrics['train_time']),
                      np.min(metrics['train_time']),
                      metrics['train_time'][-1],
                      np.sum(metrics['train_time'])))
            print(('%s %s %s %s %s Testing time:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Fore.MAGENTA + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now, string_accu,
                      np.max(metrics['test_time']),
                      np.mean(metrics['test_time']),
                      np.min(metrics['test_time']),
                      metrics['test_time'][-1],
                      np.sum(metrics['test_time'])))
            print(('%s %s %s %s CR Source:' + Fore.GREEN + ' %f%% ' + Back.BLUE + Fore.YELLOW + Style.BRIGHT + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_rate_source']) * 100,
                      np.mean(metrics['classification_rate_source']) * 100,
                      np.min(metrics['classification_rate_source']) * 100,
                      metrics['classification_rate_source'][-1] * 100))
            print(('%s %s %s %s CR Target:' + Fore.GREEN + ' %f%% ' + Back.RED + Fore.YELLOW + Style.BRIGHT + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_rate_target']) * 100,
                      np.mean(metrics['classification_rate_target']) * 100,
                      np.min(metrics['classification_rate_target']) * 100,
                      metrics['classification_rate_target'][-1] * 100))
            print(('%s %s %s %s AGMM Source:' + Fore.GREEN + ' %d ' + Fore.YELLOW + '%f' + Style.RESET_ALL + Fore.RED + ' %d' + Fore.BLUE + ' %d' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['agmm_source_size_by_batch']),
                      np.mean(metrics['agmm_source_size_by_batch']),
                      np.min(metrics['agmm_source_size_by_batch']),
                      metrics['agmm_source_size_by_batch'][-1]))
            print(('%s %s %s %s AGMM Target:' + Fore.GREEN + ' %d ' + Fore.YELLOW + '%f' + Style.RESET_ALL + Fore.RED + ' %d' + Fore.BLUE + ' %d' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['agmm_target_size_by_batch']),
                      np.mean(metrics['agmm_target_size_by_batch']),
                      np.min(metrics['agmm_target_size_by_batch']),
                      metrics['agmm_target_size_by_batch'][-1]))
            print(('%s %s %s %s Classification Source Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_source_loss']),
                      np.mean(metrics['classification_source_loss']),
                      np.min(metrics['classification_source_loss']),
                      metrics['classification_source_loss'][-1]))
            print(('%s %s %s %s Classification Target Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_target_loss']),
                      np.mean(metrics['classification_target_loss']),
                      np.min(metrics['classification_target_loss']),
                      metrics['classification_target_loss'][-1]))
            print(('%s %s %s %s Reconstruction Target Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['reconstruction_target_loss']),
                      np.mean(metrics['reconstruction_target_loss']),
                      np.min(metrics['reconstruction_target_loss']),
                      metrics['reconstruction_target_loss'][-1]))
            print(('%s %s %s %s Kullback-Leibler Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['kl_loss']),
                      np.mean(metrics['kl_loss']),
                      np.min(metrics['kl_loss']),
                      metrics['kl_loss'][-1]))
            print(('%s %s %s %s Nodes:' + Fore.GREEN + ' %d' + Fore.YELLOW + ' %f' + Fore.RED + ' %d' + Fore.BLUE + ' %d' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['node_evolution']),
                      np.mean(metrics['node_evolution']),
                      np.min(metrics['node_evolution']),
                      metrics['node_evolution'][-1]))
            print(('Network structure:' + Fore.BLUE + ' %s (Discriminative) %s (Generative)' + Style.RESET_ALL) % (
                      " ".join(map(str, nn.layers)),
                      " ".join(map(str, ae.layers))))
        print(Style.RESET_ALL)

    metrics = {'classification_rate_source': [],
               'classification_rate_target': [],
               'train_time': [],
               'test_time': [],
               'node_evolution': [],
               'classification_target_loss': [],
               'classification_source_loss': [],
               'reconstruction_source_loss': [],
               'reconstruction_target_loss': [],
               'kl_loss': [],
               'agmm_target_size_by_batch': [],
               'agmm_source_size_by_batch': [],
               'start_execution_time': time.time()}

    TorchDevice.instance().device = device

    dm = DataManipulator('')
    dm.load_custom_csv()
    dm.normalize()

    dm.split_as_source_target_streams(n_batch, 0.5)

    nn = NeuralNetwork([dm.number_features, 1, dm.number_classes])
    ae = DenoisingAutoEncoder([nn.layers[0], nn.layers[1], nn.layers[0]])

    # I am building the greedy_layer_bias
    x = dm.get_Xs(0)
    x = torch.tensor(np.atleast_2d(x), dtype=torch.float, device=MyDevice().get())
    ae.greedy_layer_wise_pretrain(x=x, number_epochs=0)
    # I am building the greedy_layer_bias

    agmm_source_discriminative = AGMM()
    agmm_target_generative = AGMM()

    for i in range(dm.number_minibatches):
        Xs = torch.tensor(dm.get_Xs(i), dtype=torch.float, device=MyDevice().get())
        ys = torch.tensor(dm.get_ys(i), dtype=torch.float, device=MyDevice().get())
        Xt = torch.tensor(dm.get_Xt(i), dtype=torch.float, device=MyDevice().get())
        yt = torch.tensor(dm.get_yt(i), dtype=torch.float, device=MyDevice().get())

        if i > 0:
            metrics['test_time'].append(time.time())
            test(nn, Xt, yt, is_source=False, is_discriminative=True, metrics=metrics)
            metrics['test_time'][-1] = time.time() - metrics['test_time'][-1]

            test(nn, Xs, ys, is_source=True, is_discriminative=True, metrics=metrics)
            test(ae, Xt, is_source=False, is_discriminative=False, metrics=metrics)

        metrics['train_time'].append(time.time())
        for epoch in range(epochs):
            for x, y in [(x.view(1, x.shape[0]), y.view(1, y.shape[0])) for x, y in zip(Xs, ys)]:
                width_evolution(network=nn, x=x, y=y, agmm=agmm_source_discriminative, train_agmm=True if epoch == 0 else False)
                if not grow_nodes(nn, ae): prune_nodes(nn, ae)
                discriminative(network=nn, x=x, y=y, agmm=agmm_source_discriminative)

            copy_weights(source=nn, target=ae, layer_numbers=[1])

            for x in [x.view(1, x.shape[0]) for x in Xt]:
                width_evolution(network=ae, x=x, agmm=agmm_target_generative, train_agmm=True if epoch == 0 else False)
                if not grow_nodes(ae, nn): prune_nodes(ae, nn)
                generative(network=ae, x=x, agmm=agmm_target_generative)

            metrics['kl_loss'].append(kl(ae=ae, x_source=Xs, x_target=Xt))
            copy_weights(source=ae, target=nn, layer_numbers=[1])

        if agmm_target_generative.M() > 1: agmm_target_generative.delete_cluster()
        if agmm_source_discriminative.M() > 1: agmm_source_discriminative.delete_cluster()

        metrics['agmm_target_size_by_batch'].append(agmm_target_generative.M())
        metrics['agmm_source_size_by_batch'].append(agmm_source_discriminative.M())
        metrics['train_time'][-1] = time.time() - metrics['train_time'][-1]
        metrics['node_evolution'].append(nn.layers[1])
        print_metrics(i + 1, metrics, nn, ae, Xs, Xt)

    result = '%f (T) ''| %f (S) \t %f | %d \t %f | %f' % (
        np.mean(metrics['classification_rate_target']),
        np.mean(metrics['classification_rate_source']),
        np.mean(metrics['node_evolution']),
        metrics['node_evolution'][-1],
        np.mean(metrics['train_time']),
        np.sum(metrics['train_time']))

    print(result)

    plot_time(metrics['train_time'], metrics['test_time'])
    plot_node_evolution(metrics['node_evolution'])
    plot_classification_rates(metrics['classification_rate_source'], metrics['classification_rate_target'])

    return result

atl = ATL(epochs = 1, device='cpu')
print(atl)

