import os
import cv2
import time
import pickle
import random
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.interpolate import LinearNDInterpolator

def grid_sample(input, grid, canvas = None):
    if canvas is None:
        output = F.grid_sample(input, grid)
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        return output, output_mask
    else:
        output = F.grid_sample(input, grid)
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output, output_mask

def folding(S, SS, grid_size_x, grid_size_y):
    ############################################# folding ##################################################
    alpha = 0.8
    multi_flag = 0
    single_multi_i = random.random()
    if single_multi_i >= 0.8:
        #print('folding single')
        v_ratio_1 = 0.3
        v_ratio_2 = 0.6
        col_row_random_i = random.random()
        if col_row_random_i <= v_ratio_1:   # row
            v1 = random.uniform(0.10, 0.25)
            v2 = 0.0
            v_i = random.random()
            if v_i > 0.5:
                v1 = v1 * -1
            v = np.array([v1, v2])
            point_i = np.random.randint(int(grid_size_x * (grid_size_y / 8)), int(grid_size_x * (grid_size_y * 7 / 8)), 1)
            q = S[point_i]
        elif col_row_random_i <= v_ratio_2:   # column
            v1 = 0.0
            v2 = random.uniform(0.10, 0.25)
            v_i = random.random()
            if v_i > 0.5:
                v2 = v2 * -1
            v = np.array([v1, v2])
            point_i = np.random.randint(int(grid_size_x * (grid_size_y / 8)), int(grid_size_x * (grid_size_y * 7 / 8)), 1)
            q = SS[point_i]
        else:                                   # oblique
            v1 = random.uniform(0.05, 0.28)
            v2 = random.uniform(0.05, 0.28)
            v_i = random.random()
            if v_i > 0.5:
                v1 = v1 * -1
            v_i = random.random()
            if v_i > 0.5:
                v2 = v2 * -1
            v = np.array([v1, v2])
            point_i = np.random.randint(0, grid_size_x * grid_size_y - 1, 1)
            q = S[point_i]

        qp = S - q
        d = np.linalg.norm(np.cross(qp, v).reshape((grid_size_x * grid_size_y, 1)) / np.linalg.norm(v), axis=1, keepdims=True)
        w = alpha / (d + alpha)
        T = S + w * v

    else:
        #print('folding multi')
        multi_flag = 1
        v_ratio_1_m = 0.15
        v_ratio_2_m = 0.3
        v_ratio_3_m = 0.8
        wv = 0
        point_ratio = 4
        num_n = random.sample(range(2, 9), 1)
        col_row_random_i = random.random()
        if col_row_random_i <= v_ratio_1_m:  # multi_row
            row_z = 0
            row_f = 0
            for n_i in range(num_n[0]):
                v1 = random.uniform(0.10, 0.15)
                v2 = 0.0
                v_i = random.random()
                if v_i > 0.5:
                    v1 = v1 * -1
                if v1 >= 0:
                    row_z += 1
                else:
                    row_f += 1
                if row_z >= point_ratio:
                    v1 = -abs(v1)
                    row_z -= 1
                    row_f += 1
                elif row_f >= point_ratio:
                    v1 = abs(v1)
                    row_f -= 1
                    row_z += 1
                else:
                    pass
                v = np.array([v1, v2])
                point_i = np.random.randint(int(grid_size_x * (grid_size_y / 8)),
                                            int(grid_size_x * (grid_size_y * 7 / 8)),
                                            1)
                q = S[point_i]
                qp = S - q
                d = np.linalg.norm(np.cross(qp, v).reshape((grid_size_x * grid_size_y, 1)) / np.linalg.norm(v), axis=1, keepdims=True)
                w = alpha / (d + alpha)
                wv += w * v
        elif col_row_random_i <= v_ratio_2_m:  # multi_column
            col_z = 0
            col_f = 0
            for n_i in range(num_n[0]):
                v1 = 0.0
                v2 = random.uniform(0.10, 0.15)
                v_i = random.random()
                if v_i > 0.5:
                    v2 = v2 * -1
                if v2 >= 0:
                    col_z += 1
                else:
                    col_f += 1
                if col_z >= point_ratio:
                    v2 = -abs(v2)
                    col_z -= 1
                    col_f += 1
                elif col_f >= point_ratio:
                    v2 = abs(v2)
                    col_f -= 1
                    col_z += 1
                else:
                    pass
                v = np.array([v1, v2])
                point_i = np.random.randint(int(grid_size_x * (grid_size_y / 8)),
                                            int(grid_size_x * (grid_size_y * 7 / 8)),
                                            1)
                q = SS[point_i]
                qp = S - q
                d = np.linalg.norm(np.cross(qp, v).reshape((grid_size_x * grid_size_y, 1)) / np.linalg.norm(v), axis=1, keepdims=True)
                w = alpha / (d + alpha)
                wv += w * v
        elif col_row_random_i <= v_ratio_3_m:  # multi_oblique
            for n_i in range(num_n[0]):
                v1 = random.uniform(0.05, 0.28)
                v2 = random.uniform(0.05, 0.28)
                v_i = random.random()
                if v_i > 0.5:
                    v1 = v1 * -1
                v_i = random.random()
                if v_i > 0.5:
                    v2 = v2 * -1
                v = np.array([v1, v2])
                point_i = np.random.randint(0, grid_size_x * grid_size_y - 1, 1)
                q = S[point_i]
                qp = S - q
                d = np.linalg.norm(np.cross(qp, v).reshape((grid_size_x * grid_size_y, 1)) / np.linalg.norm(v), axis=1,
                                   keepdims=True)
                w = alpha / (d + alpha)
                wv += w * v
        else: # multi_row_column_oblique
            v_ratio_1_mm = 0.3
            v_ratio_2_mm = 0.6
            for n_i in range(num_n[0]):
                col_row_random_i = random.random()
                if col_row_random_i <= v_ratio_1_mm:
                    v1 = random.uniform(0.10, 0.25)
                    v2 = 0.0
                    v_i = random.random()
                    if v_i > 0.5:
                        v1 = v1 * -1
                    v = np.array([v1, v2])
                    point_i = np.random.randint(int(grid_size_x * (grid_size_y / 8)),
                                                int(grid_size_x * (grid_size_y * 7 / 8)), 1)
                    q = S[point_i]
                    qp = S - q
                    d = np.linalg.norm(np.cross(qp, v).reshape((grid_size_x * grid_size_y, 1)) / np.linalg.norm(v),
                                       axis=1,
                                       keepdims=True)
                    w = alpha / (d + alpha)
                    wv += w * v

                elif col_row_random_i <= v_ratio_2_mm:
                    v1 = 0.0
                    v2 = random.uniform(0.10, 0.25)
                    v_i = random.random()
                    if v_i > 0.5:
                        v2 = v2 * -1
                    v = np.array([v1, v2])
                    point_i = np.random.randint(int(grid_size_x * (grid_size_y / 8)),
                                                int(grid_size_x * (grid_size_y * 7 / 8)), 1)
                    q = SS[point_i]
                    qp = S - q
                    d = np.linalg.norm(np.cross(qp, v).reshape((grid_size_x * grid_size_y, 1)) / np.linalg.norm(v),
                                       axis=1,
                                       keepdims=True)
                    w = alpha / (d + alpha)
                    wv += w * v
                else:
                    v1 = random.uniform(0.05, 0.28)
                    v2 = random.uniform(0.05, 0.28)
                    v_i = random.random()
                    if v_i > 0.5:
                        v1 = v1 * -1
                    v_i = random.random()
                    if v_i > 0.5:
                        v2 = v2 * -1
                    v = np.array([v1, v2])
                    point_i = np.random.randint(0, grid_size_x * grid_size_y - 1, 1)
                    q = S[point_i]
                    qp = S - q
                    d = np.linalg.norm(np.cross(qp, v).reshape((grid_size_x * grid_size_y, 1)) / np.linalg.norm(v),
                                       axis=1,
                                       keepdims=True)
                    w = alpha / (d + alpha)
                    wv += w * v

        T = S + wv
    return T, multi_flag

def curving(S, SS, grid_size_x, grid_size_y):
    ############################################# curving ##################################################
    alpha = 2
    v_ratio_1 = 0.33
    v_ratio_2 = 0.65
    col_row_random_i = random.random()
    if col_row_random_i <= v_ratio_1:   # row
        v1 = random.uniform(0.10, 0.15)
        v2 = 0.0
        v_i = random.random()
        if v_i > 0.5:
            v1 = v1 * -1
        v = np.array([v1, v2])
        point_i = np.random.randint(int(grid_size_x * (grid_size_y / 8)), int(grid_size_x * (grid_size_y * 7 / 8)),
                                    1)
        q = S[point_i]
    elif col_row_random_i <= v_ratio_2:   # column
        v1 = 0.0
        v2 = random.uniform(0.10, 0.15)
        v_i = random.random()
        if v_i > 0.5:
            v2 = v2 * -1
        v = np.array([v1, v2])
        point_i = np.random.randint(int(grid_size_x * (grid_size_y / 8)), int(grid_size_x * (grid_size_y * 7 / 8)),
                                    1)
        q = SS[point_i]
    else:                                        # oblique
        v1 = random.uniform(0.05, 0.15)
        v2 = random.uniform(0.05, 0.15)
        v_i = random.random()
        if v_i > 0.5:
            v1 = v1 * -1
        v_i = random.random()
        if v_i > 0.5:
            v2 = v2 * -1
        v = np.array([v1, v2])
        point_i = np.random.randint(0, grid_size_x * grid_size_y - 1, 1)
        q = S[point_i]

    qp = S - q
    d = np.linalg.norm(np.cross(qp, v).reshape((grid_size_x * grid_size_y, 1)) / np.linalg.norm(v), axis=1,
                       keepdims=True)
    w = 1 - (d ** alpha)
    T = S + w * v
    return T

def HSV_func(image_disorted):
    image_tmp = image_disorted[0].data.numpy()
    image_tmp = image_tmp.transpose(1, 2, 0)
    HSV = cv2.cvtColor(image_tmp, cv2.COLOR_BGR2HSV)
    #H, S, V = cv2.split(HSV)
    H = HSV[:, :, 0]
    S = HSV[:, :, 1]
    V = HSV[:, :, 2]
    h_random = random.random() * 10
    s_random = (random.random() * 2 - 1) * 0.025
    random_i = random.random()
    if random_i >= 0.42:
        v_random = random.random() * 50 - 50
    else:
        v_random = random.random() * 30
    H = H + h_random   # + 10
    S = S + s_random   # (-0.025, 0.025)
    V = V + v_random   # (-50, 30)
    id_360 = H > 360
    H[id_360] = 360
    id_0 = H < 0
    H[id_0] = 0
    id_1 = S > 1
    S[id_1] = 1
    id_0 = S < 0
    S[id_0] = 0
    id_255 = V > 255
    V[id_255] = 255
    id_0 = V < 0
    V[id_0] = 0
    HSV[:, :, 0] = H
    HSV[:, :, 1] = S
    HSV[:, :, 2] = V
    image_return = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    image_return = torch.Tensor(image_return[np.newaxis, :, :, :].transpose(0, 3, 1, 2))

    return image_return



def get_traindata():
    root_img = './data_source'
    save_dir = './image_save_for_reference'
    data_save = './data_save'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(data_save):
        os.makedirs(data_save)
    root_bg = './bg'

    ####################################################################################################################
    r1 = 0.95
    r2 = 0.95
    c1 = 0.95
    c2 = 0.95
    c11 = 0.95
    c22 = 0.95
    grid_size_x = 128
    grid_size_y = 128
    doc_size_x = 512
    doc_size_y = 512
    batch_size_process_data = 1
    cycle_img = 2
    save_docimage = True
    folging_flag = 0
    curving_flag = 0
    folding_iter = 0
    curving_iter = 0
    ####################################################################################################################
    S = torch.Tensor(list(itertools.product(
        np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_size_x - 1)),
        np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_size_y - 1)),
    )))
    Y, X = S.split(1, dim=1)
    S = torch.cat([X, Y], dim=1)
    S = S.numpy()

    SS = S.reshape(grid_size_x, grid_size_y, 2)
    SS = SS.transpose([1, 0, 2])
    SS = SS.reshape(grid_size_x * grid_size_y, 2)

    coord_S = torch.Tensor(list(itertools.product(
        np.arange(-c1, c1 + 0.00001, 2.0 * c1 / (doc_size_x - 1)),
        np.arange(-c2, c2 + 0.00001, 2.0 * c2 / (doc_size_y - 1)),
    )))
    Y, X = coord_S.split(1, dim=1)
    coord_S = torch.cat([X, Y], dim=1)
    coord_S = coord_S.numpy()

    coord_T = torch.Tensor(list(itertools.product(
        np.arange(-c11, c11 + 0.00001, 2.0 * c11 / (doc_size_x - 1)),
        np.arange(-c22, c22 + 0.00001, 2.0 * c22 / (doc_size_y - 1)),
    )))
    Y, X = coord_T.split(1, dim=1)
    coord_T = torch.cat([X, Y], dim=1)
    coord_T = coord_T.numpy()
    ####################################################################################################################
    img_bg_list = os.listdir(root_bg)
    img_list = os.listdir(root_img)
    all = time.time()
    img_idx = 0
    for i in range(len(img_list)):
        img = cv2.imread(root_img + '/' + img_list[i])
        img_resize = cv2.resize(img, (doc_size_x, doc_size_y))
        img_h = img_resize.shape[0]
        img_w = img_resize.shape[1]
        img_resize = img_resize.transpose(2, 0, 1)
        img_resize = img_resize[np.newaxis, :, :, :]
        img_resize = torch.Tensor(img_resize)
        for i_cycle in range(cycle_img):
            output_datasets = []
            i_bg = random.randint(0, len(img_bg_list) - 1)
            img_bg = cv2.imread(root_bg + '/' + img_bg_list[i_bg])
            img_bg = cv2.resize(img_bg, (img_w, img_h))
            img_bg = img_bg.transpose(2, 0, 1)
            img_bg = img_bg[np.newaxis, :, :, :]
            img_bg_ = torch.Tensor(img_bg)  # torch.size([1, 3, 512, 512])
            ############################################################################################################
            w_i = random.random()
            multi_flag = 0
            if w_i >= 0.5:
                ### folding ###
                T, multi_flag = folding(S, SS, grid_size_x, grid_size_y)
                folging_flag = 1
                folding_iter += 1
            else:
                ### curving ###
                T = curving(S, SS, grid_size_x, grid_size_y)
                curving_flag = 1
                curving_iter += 1
            T_transpose = T.transpose().copy()
            x = T_transpose[0]
            y = T_transpose[1]
            x_min = np.min(x)
            x_max = np.max(x)
            y_min = np.min(y)
            y_max = np.max(y)
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            T_transpose[0] -= x_center
            T_transpose[1] -= y_center

            x = T_transpose[0]
            y = T_transpose[1]
            x_min = np.min(x)
            x_max = np.max(x)
            y_min = np.min(y)
            y_max = np.max(y)
            xy_ratio = max(abs(x_max), abs(y_max), abs(x_min), abs(y_min))
            if folging_flag == 1:
                if multi_flag == 1:
                    T_transpose = T_transpose * xy_ratio * 1.15
                else:
                    T_transpose = T_transpose * xy_ratio * 1.1
                folging_flag = 0
            elif curving_flag == 1:
                T_transpose = T_transpose * xy_ratio * 1.2
                curving_flag = 0
            else:
                T_transpose = T_transpose * xy_ratio * 1.2
                folging_flag = 0
                curving_flag = 0
            T = T_transpose.transpose()

            ############################################################################################################
            transfer_S2T = LinearNDInterpolator(S, T)
            grid_T = transfer_S2T(coord_S)
            grid_T = torch.Tensor(grid_T)
            grid_T = grid_T.view(batch_size_process_data, img_h, img_w, 2)
            image_disorted, _ = grid_sample(img_resize, grid_T, img_bg_)
            random_hsv = random.random()
            if random_hsv > 0.8:
                image_disorted = HSV_func(image_disorted)
            transfer_T2S = LinearNDInterpolator((1 / r1) * T, (1 / r1) * S)
            grid_S = transfer_T2S(coord_T * (1 / r1))
            where_are_nan = np.isnan(grid_S)
            grid_S[where_are_nan] = -2.0
            ##
            grid_S_ = torch.Tensor(grid_S)  # regress it
            grid_S_ = grid_S_.view(batch_size_process_data, img_h, img_w, 2)
            image_cycle, _ = grid_sample(image_disorted, grid_S_)
            ############################################################################################################
            image_save = img_resize.data.numpy()
            image_save = np.uint8(image_save.transpose(0, 2, 3, 1))
            image_disorted_save = image_disorted.data.numpy()
            image_disorted_save = np.uint8(image_disorted_save.transpose(0, 2, 3, 1))
            image_cycle_save = image_cycle.data.numpy()
            image_cycle_save = np.uint8(image_cycle_save.transpose(0, 2, 3, 1))
            grid_S_save = np.float32(grid_S)

            if save_docimage:
                cv2.imwrite(save_dir + '/{}_{}_ori.jpg'.format(i, i_cycle), image_save[0])
                cv2.imwrite(save_dir + '/{}_{}_disorted.jpg'.format(i, i_cycle), image_disorted_save[0])
                cv2.imwrite(save_dir + '/{}_{}_cycle.jpg'.format(i, i_cycle), image_cycle_save[0])
            output_datasets.append({
                'img_ori': image_save[0],
                'img': image_disorted_save[0],
                'grid_back': grid_S_save
            })
            save_name = data_save + '/%05d.pkl' % (img_idx)
            img_idx += 1
            with open(save_name, 'wb') as f_out:
                pickle.dump(output_datasets, f_out)
            print('save train data to: ', save_name)


        if i % 100 == 0:
            print('iter:', i)
            print('folding number:', folding_iter)
            print('curving number:', curving_iter)
    print('Time cost:', time.time()-all)

if __name__ == '__main__':
    get_traindata()

