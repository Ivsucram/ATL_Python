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
#
from MyUtil import MyUtil
from MySingletons import MyDevice

import numpy as np
import torch
import math


class GMM:
    weight = 1
    center = None
    variance = None
    win_counter = 1
    inference_sum = 0
    survive_counter = 0
    y_count = None
    dist = None

    _number_features = None

    @property
    def inference(self):
        if self.dist is not None:
            return torch.min(self.dist)
        return torch.tensor(0)

    @property
    def hyper_volume(self):
        if self.dist is not None:
            return self.variance[0][torch.argmin(self.dist)]
        return torch.tensor([[0.01]])

    @property
    def standard_deviation(self):
        return torch.sqrt(self.variance)

    @property
    def std(self):
        return self.standard_deviation

    @property
    def precision(self):
        return 1/self.variance

    @property
    def first_order_moment(self):
        return self.center

    @property
    def second_order_moment(self):
        return self.center ** 2 + self.variance

    @property
    def mode(self):
        return self.center

    @property
    def probX_J(self):
        denumerator = torch.sqrt(2 * math.pi * self.hyper_volume)
        return denumerator * self.inference

    def __init__(self, x):
        self._number_features = x.shape[1]
        self.center = x
        self.variance = 0.01 + torch.zeros(int(self._number_features), dtype=torch.float, device=MyDevice().get())
        self.variance = self.variance.view(1, self.variance.shape[0])

    def compute_inference(self, x, y=None):
        if y is not None:
            if self.y_count is None:
                self.y_count = y
            else:
                self.y_count += y

        c = self.center

        dist = ((x - c) ** 2) / (- 2 * self.variance)
        self.dist = torch.exp(dist)


class AGMM:
    gmm_array = None
    number_samples_feed = 0
    rho = 0.1
    number_features = None

    @property
    def hyper_volume(self):
        hyper_volume = torch.tensor(0, dtype=torch.float, device=MyDevice().get())
        for i in range(len(self.gmm_array)):
            hyper_volume += self.gmm_array[i].hyper_volume
        return hyper_volume

    @property
    def weight_sum(self):
        weight_sum = torch.tensor(0, dtype=torch.float, device=MyDevice().get())
        for i in range(self.M()):
            weight_sum += self.gmm_array[i].weight
        return weight_sum

    def run(self, x, bias2):
        self.number_samples_feed += 1
        if self.gmm_array is None:
            self.gmm_array = [GMM(x)]
            self.number_features = x.shape[1]
        else:
            self.compute_inference(x)
            gmm_winner_idx = np.argmax(self.update_weights())

            if self.M() > 1:
                self.compute_overlap_degree(gmm_winner_idx, 3, 3)

            denominator = 1.25 * torch.exp(-bias2) + 0.75 * self.number_features
            numerator = 4 - 2 * torch.exp(torch.tensor(-self.number_features / 2.0, dtype=torch.float, device=MyDevice().get()))
            threshold = torch.exp(- denominator / numerator)

            condition1 = self.gmm_array[gmm_winner_idx].inference < threshold
            condition2 = self.gmm_array[gmm_winner_idx].hyper_volume > self.rho * (self.hyper_volume - self.gmm_array[gmm_winner_idx].hyper_volume)
            condition3 = self.number_samples_feed > 10
            if condition1 and condition2 and condition3:
                self.create_cluster(x)
                self.gmm_array[-1].variance = (x - self.gmm_array[gmm_winner_idx].center) ** 2
            else:
                self.update_cluster(x, self.gmm_array[gmm_winner_idx])

    def create_cluster(self, x):
        self.gmm_array.append(GMM(x))

        weight_sum = self.weight_sum
        for gmm in self.gmm_array:
            gmm.weight = gmm.weight / weight_sum

    def update_cluster(self, x, gmm):
        gmm.win_counter += 1
        gmm.center += (x - gmm.center) / gmm.win_counter
        gmm.variance += ((x - gmm.center) ** 2 - gmm.variance) / gmm.win_counter

    def delete_cluster(self):
        if self.M() <= 1:
            return

        accumulated_inference = []
        for gmm in self.gmm_array:
            if gmm.survive_counter > 0:
                accumulated_inference.append(gmm.inference_sum / gmm.survive_counter)

        accumulated_inference = torch.stack(accumulated_inference)[torch.isnan(torch.stack(accumulated_inference)) == False]

        delete_list = torch.where(accumulated_inference <= (torch.mean(accumulated_inference) - 0.5 * torch.std(accumulated_inference)))[0]
        if len(delete_list) == len(self.gmm_array):
            raise TypeError('problem')  # FIXME if this happen, it means you have a great problem at your code

        if len(delete_list):
            self.gmm_array = np.delete(self.gmm_array, delete_list.cpu().numpy()).tolist()
            accumulated_inference = torch.tensor(np.delete(accumulated_inference.cpu().numpy(), delete_list.cpu().numpy()).tolist(), dtype=torch.float, device=MyDevice().get())

        sum_weight = 0
        for gmm in self.gmm_array:
            sum_weight += gmm.weight

        if sum_weight == 0:
            max_index = torch.argmax(accumulated_inference)
            self.gmm_array[max_index].weight += 1
            sum_weight = 0
            for gmm in self.gmm_array:
                sum_weight += gmm.weight

        for gmm in self.gmm_array:
            gmm.weight = gmm.weight / sum_weight

    def compute_inference(self, x, y=None):
        for gmm in self.gmm_array:
            gmm.compute_inference(x, y)

    def update_weights(self):
        probX_JprobJ = torch.zeros(len(self.gmm_array))
        weights = torch.zeros(len(self.gmm_array))

        sum_winner_counter = 0
        max_inference = 0
        max_inference_index = 0
        for i in range(self.M()):
            sum_winner_counter += self.gmm_array[i].win_counter
            if self.gmm_array[i].inference > max_inference:
                max_inference = self.gmm_array[i].inference
                max_inference_index = i

        for i in range(self.M()):
            self.gmm_array[i].inference_sum += self.gmm_array[i].inference
            self.gmm_array[i].survive_counter += 1

            probJ = self.gmm_array[i].win_counter / sum_winner_counter
            probX_JprobJ[i] = self.gmm_array[i].probX_J * probJ

        if torch.sum(probX_JprobJ) == 0:
            probX_JprobJ[max_inference_index] += 1

        for i in range(self.M()):
            self.gmm_array[i].weight = float(probX_JprobJ[i] / torch.sum(probX_JprobJ))
            weights[i] = self.gmm_array[i].weight

        return weights

    def compute_overlap_degree(self, gmm_winner_idx, maximum_limit=None, minimum_limit=None):
        if maximum_limit is None:
            maximum_limit = minimum_limit = 3
        elif minimum_limit is None:
            minimum_limit = maximum_limit

        overlap_coefficient = torch.tensor(1 / self.M(), dtype=torch.float, device=MyDevice().get())

        sigma_maximum_winner = maximum_limit * torch.sqrt(self.gmm_array[gmm_winner_idx].variance)
        sigma_minimum_winner = minimum_limit * torch.sqrt(self.gmm_array[gmm_winner_idx].variance)

        winner_center = self.gmm_array[gmm_winner_idx].center

        if maximum_limit == minimum_limit:
            mean_positive_sigma_winner = winner_center + sigma_maximum_winner
            mean_negative_sigma_winner = winner_center - sigma_minimum_winner
        else:
            # FIXME This seems wrong
            mean_positive_sigma_winner = winner_center + sigma_minimum_winner + sigma_maximum_winner
            mean_negative_sigma_winner = winner_center - sigma_minimum_winner - sigma_maximum_winner

        mean_positive_sigma = torch.zeros(int(self.M()), int(self.number_features), dtype=torch.float, device=MyDevice().get())
        mean_negative_sigma = torch.zeros(int(self.M()), int(self.number_features), dtype=torch.float, device=MyDevice().get())
        overlap_mins_mins   = torch.zeros(int(self.M()), int(self.number_features), dtype=torch.float, device=MyDevice().get())
        overlap_mins_plus   = torch.zeros(int(self.M()), int(self.number_features), dtype=torch.float, device=MyDevice().get())
        overlap_plus_mins   = torch.zeros(int(self.M()), int(self.number_features), dtype=torch.float, device=MyDevice().get())
        overlap_plus_plus   = torch.zeros(int(self.M()), int(self.number_features), dtype=torch.float, device=MyDevice().get())

        overlap_score = []

        for i in range(self.M()):
            sigma_maximum = maximum_limit * torch.sqrt(self.gmm_array[i].variance)
            sigma_minimum = minimum_limit * torch.sqrt(self.gmm_array[i].variance)

            if maximum_limit == minimum_limit:
                mean_positive_sigma[i] = self.gmm_array[i].center + sigma_maximum
                mean_negative_sigma[i] = self.gmm_array[i].center - sigma_maximum
            else:
                #FIXME This seems wrong
                mean_positive_sigma[i] = sigma_maximum + sigma_minimum
                mean_negative_sigma[i] = -sigma_minimum - sigma_maximum

            overlap_mins_mins[i] = torch.mean(mean_negative_sigma[i] - mean_negative_sigma_winner)
            overlap_mins_plus[i] = torch.mean(mean_positive_sigma[i] - mean_negative_sigma_winner)
            overlap_plus_mins[i] = torch.mean(mean_negative_sigma[i] - mean_positive_sigma_winner)
            overlap_plus_plus[i] = torch.mean(mean_positive_sigma[i] - mean_positive_sigma_winner)

            condition1 = (overlap_mins_mins[i] >= 0).all() \
                     and (overlap_mins_plus[i] >= 0).all() \
                     and (overlap_plus_mins[i] <= 0).all() \
                     and (overlap_plus_plus[i] <= 0).all()
            condition2 = (overlap_mins_mins[i] <= 0).all() \
                     and (overlap_mins_plus[i] >= 0).all() \
                     and (overlap_plus_mins[i] <= 0).all() \
                     and (overlap_plus_plus[i] >= 0).all()
            condition3 = (overlap_mins_mins[i] > 0).all() \
                     and (overlap_mins_plus[i] > 0).all() \
                     and (overlap_plus_mins[i] < 0).all() \
                     and (overlap_plus_plus[i] > 0).all()
            condition4 = (overlap_mins_mins[i] < 0).all() \
                     and (overlap_mins_plus[i] > 0).all() \
                     and (overlap_plus_mins[i] < 0).all() \
                     and (overlap_plus_plus[i] < 0).all()

            if condition1 or condition2:
                # full overlap, the cluster is inside the winning cluster
                # the score is full score
                overlap_score.append(overlap_coefficient)
            elif condition3 or condition4:
                # partial overlap, the score is the full score multiplied by the overlap degree
                reward = MyUtil.norm_2(self.gmm_array[i].center - self.gmm_array[gmm_winner_idx].center) \
                       / MyUtil.norm_2(self.gmm_array[i].center + self.gmm_array[gmm_winner_idx].center) \
                       + MyUtil.norm_2(self.gmm_array[i].center - torch.sqrt(self.gmm_array[gmm_winner_idx].variance)) \
                       / MyUtil.norm_2(self.gmm_array[i].center + torch.sqrt(self.gmm_array[gmm_winner_idx].variance))
                overlap_score.append(overlap_coefficient * reward)
            else:
                # No overlap, then the score is 0
                overlap_score.append(torch.zeros(1))

        overlap_score.pop(gmm_winner_idx)
        self.rho = torch.sum(torch.stack(overlap_score))
        self.rho = torch.min(self.rho, torch.ones_like(self.rho))
        self.rho = torch.max(self.rho, torch.ones_like(self.rho) * 0.1)  # Do not let rho = zero

    def compute_rho_vigilance_test(self, x, gmm_winner_idx):
        pass

    def compute_rho_containing_rule(self, gmm_winner_idx, maximum_limit, minimum_limit):
        pass

    def compute_number_of_gmms(self):
        if self.gmm_array is None:
            return 0
        else:
            return len(self.gmm_array)

    def M(self):
        return self.compute_number_of_gmms()