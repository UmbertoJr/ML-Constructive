import os
import torch
import cv2 as cv
import numpy as np
from scipy.spatial.distance import pdist


class pixel_in_image:
    def __init__(self, max_value, num_pixel, raggio):
        self.max_value = float(max_value)
        self.num_pixel = float(num_pixel)
        self.raggio = float(raggio)

    def pixel_pos(self, vec):
        num_pixel, raggio = [self.num_pixel, self.raggio]
        max_value = self.max_value
        x = int(round((vec[0]) / max_value * (num_pixel // 2 - 2 * raggio)) + num_pixel // 2)
        y_ = round((vec[1]) / max_value * (num_pixel // 2 - 2 * raggio))
        y = int(-y_ + (num_pixel // 2))
        return x, y

    def pixel_distance(self):
        num_pixel, raggio = [self.num_pixel, self.raggio]
        max_value = self.max_value
        return max_value / (num_pixel // 2 - 2 * raggio)


def transformation(pos, city1, city2, neigs):
    centro_img = (pos[city1] + pos[city2]) / 2.
    # centro_img = pos[city1]
    # print(centro, step_prec, neigs_prec)
    # print(city1, city2, neigs)
    points_go = np.concatenate([[city1, city2], list(neigs)])
    # pos_selected_go = pos[points_go] - pos[centro]
    pos_selected_go = pos[points_go] - centro_img
    max_value = np.max(np.linalg.norm(pos_selected_go, axis=1))
    return pos_selected_go, max_value, points_go


def sort_the_list_of_files(path):
    list_files = os.listdir(path)
    dic_files = {int(f_n[:-8]): f_n for f_n in list_files if ".h5" in f_n}
    sorted_list_of_file = [el[1] for el in sorted(dic_files.items(), key=lambda kv: kv[0])]
    return sorted_list_of_file


def slice_iterator(slice):
    if not str(slice).isdigit():
        start_sl, stop_sl, step_sl = str(slice).replace("(", ",").replace(")", ",").split(',')[1:4]
        if start_sl == 'None':
            iterator_slice = range(int(stop_sl))
            if step_sl != ' None':
                assert False, "step of the iterator should be 0"
        else:
            iterator_slice = 1
            if stop_sl != "None":
                iterator_slice = range(int(start_sl), int(stop_sl))
                if step_sl != " None":
                    assert False, "step of the iterator should be 0"
    else:
        iterator_slice = [slice]

    return iterator_slice


def plot_cv(im_list, city):
    names = ['andata_primo_canale', 'andata_secondo_canale', 'andata',
             'ritorno_primo_canale', 'ritorno_primo_canale', 'ritorno',
             'posizione']
    for ind, el in enumerate(im_list):
        el = (el * 255).astype(np.uint8)
        # el = (el).astype(np.uint8)
        img = cv.resize(el, (96 * 4, 96 * 4), interpolation=cv.INTER_NEAREST)
        # img = el
        # print(img.shape)
        # if len(img.shape) > 2:
        #     if img.shape[2] == 2:
        #         img = np.stack([img[:, :, 0], img[:, :, 1], img[:, :, 0]], axis=2)
        #         cv.imwrite(f'./data/images/example/{names[ind]}_{np.random.uniform(0, 1)}.png', img)
        #     elif img.shape[2] == 3:
        #         cv.imwrite(f'./data/images/example/example{city}_{np.random.uniform(0, 1)}.png', img)
        # print(img.shape)
        # print()
        cv.imshow(names[ind], img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def plot_single_cv(image):
    el = (image * 255).astype(np.uint8)
    img = cv.resize(el, (96 * 4, 96 * 4), interpolation=cv.INTER_NEAREST)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def normalize_image(img):
    img = np.array(img, dtype=np.float)
    return img / 255.


def distance_mat(pos):
    distance = create_upper_matrix(pdist(pos, "euclidean"), pos.shape[0])
    distance = np.round((distance.T + distance) * 100000, 0) / 100000
    return distance


def create_upper_matrix(values, size):
    """
    builds an upper matrix
    @param values: to insert in the matrix
    @param size: of the matrix
    @return:
    """
    upper = np.zeros((size, size))
    r = np.arange(size)
    mask = r[:, None] < r
    upper[mask] = values
    return upper


def to_torch(input_image):
    input_images = torch.tensor(input_image, dtype=torch.float, device='cpu')
    input_images = input_images.permute(0, 3, 1, 2)
    return input_images
