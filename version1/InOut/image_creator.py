import cv2 as cv

import torch
import numpy as np
from torch import tensor

from InOut.output_agent import OutputHandler
from InOut.candidateSets import CandidatesAgent
from InOut.utils import transformation, pixel_in_image, normalize_image, plot_cv, distance_mat, plot_single_cv

intensity = 255
to_plot = False


def innerLoopTracker(edge_to_append, sol):
    n1, n2 = edge_to_append
    # print(sol[n1], sol[n2], n1, n2)
    if len(sol[n1]) > 2 or len(sol[n2]) > 2:
        return False
    if len(sol[n1]) == 0:
        return True
    if len(sol[n2]) == 0:
        return True
    cur_city = sol[n1][0]
    partial_tour = [n1, cur_city]
    while True:
        if len(sol[cur_city]) == 2:
            for i in sol[cur_city]:
                if i not in partial_tour:
                    # print(i)
                    # print(i, partial_tour, n1, n2)
                    # print(sol)
                    cur_city = i
                    partial_tour.append(cur_city)
                    if cur_city == n2:
                        return False
        else:
            return True


def create_LP(num_cit, neighborhood, dist_matrix):
    len_neig = len(neighborhood[0])
    LP_v = {i: {} for i in range(len_neig)}
    keys = []
    return_list = []
    for in_cl in range(len_neig):
        for node in range(num_cit):
            h = neighborhood[node][in_cl]
            if (node, h) not in keys and (h, node) not in keys:
                LP_v[in_cl][(node, h)] = dist_matrix[node, h]
                keys.append((node, h))

    for in_cl in range(len_neig):
        return_list.extend([k for k, v in sorted(LP_v[in_cl].items(), key=lambda item: item[1])])

    return return_list


def create_neigs(num_cit, dist_matrix, k):
    neigs = {}
    for i in range(num_cit):
        neigs[i] = np.argsort(dist_matrix[i])[1: k + 1]
    return neigs


class ImageTrainDataCreator:
    """
    class in charge to create the input images and output.
    The output is True if the input draws an optimal edge.
    """

    def __init__(self, settings, cases=1):
        self.settings = settings
        self.create_image = ImageCreator(settings)
        self.input_channels = 3
        self.cases = cases
        self.type_output = torch.long
        if to_plot:
            self.create_in_out = output_visual_checker(settings, self.create_in_out)

    def get_num_of_images(self, number_cities, pos):
        dist_matrix = distance_mat(pos)
        LP = create_LP(number_cities, create_neigs(number_cities, dist_matrix, self.settings.cases_in_L_P), dist_matrix)
        # print(f"number cities :{number_cities}")
        partial_sol = {i: [] for i in range(number_cities)}
        count = 0
        for city1, city2 in LP:
            if innerLoopTracker((city1, city2), partial_sol):
                count += 1
                partial_sol[city1].append(city2)
                partial_sol[city2].append(city1)
        return count

    def create_data_for_all(self, data):
        number_cities, pos, tour = data
        dist_matrix = distance_mat(pos)
        LP = create_LP(number_cities, create_neigs(number_cities, dist_matrix, self.settings.cases_in_L_P), dist_matrix)
        num_images = len(LP)
        return self.create_data(num_images, pos, LP, tour, number_cities)

    def create_data(self, num_images, pos, LP, optimal_tour, number_cities):
        settings = self.settings

        # creation empty data collector
        input_images, output_data = self.create_empty_collector(num_images, settings)

        # tools initialization
        dist_matrix, pos, max_global, candidates_agent, output_handler = self.tools_init(pos, optimal_tour)
        partial_sol = {i: [] for i in range(number_cities)}

        # selects the list of promising edges
        iter_ = 0
        # print(len(LP), )
        for city1, city2 in LP:
            if innerLoopTracker((city1, city2), partial_sol):
                # select the candidate set for the current edge l=[city1, city2]
                neig = candidates_agent.create_candidate(city1, city2)

                # the fun creates input and output for current city
                image_, out_ = self.create_in_out(city1, city2, neig, pos, output_handler, partial_sol)

                if out_ == 1:
                    partial_sol[city1].append(city2)
                    partial_sol[city2].append(city1)

                output_data[iter_] = out_
                input_images[iter_] = image_
                iter_ += 1

        input_images, output_data = self.to_torch(input_images, output_data)
        return input_images, output_data

    def to_torch(self, input_images, output_data):
        input_images = tensor(input_images, dtype=torch.float)
        input_images = input_images.permute(0, 3, 1, 2)
        output_ = tensor(output_data, dtype=self.type_output)
        return input_images, output_

    def create_in_out(self, city, city2, neig, pos, output_handler, p_sol):
        # return outputs (angles or neig index) for the current city (go and come)
        out_ = output_handler.create_output(city, city2)
        # create image
        image_ = self.create_image(city, city2, pos, neig, p_sol)
        return image_, out_

    def create_empty_collector(self, num_images, settings):
        input_images = np.zeros((num_images, settings.num_pixels, settings.num_pixels, 3), np.uint8)
        output_ = np.zeros((num_images), np.int)
        return input_images, output_

    def tools_init(self, pos, optimal_tour):
        # distance matrix creation, problem centering and Pixel Handler Agent
        settings = self.settings
        dist_matrix = distance_mat(pos)
        pos -= np.mean(pos, axis=0)
        max_global = np.max(np.linalg.norm(pos, axis=1))

        # preprocessing for candidates
        candidates_agent = CandidatesAgent(settings, dist_matrix)

        # init the output handler
        output_handler = OutputHandler(settings, optimal_tour)
        return dist_matrix, pos, max_global, candidates_agent, output_handler


class ImageCreator:
    def __init__(self, settings):
        self.settings = settings
        self.num_pixel = settings.num_pixels
        self.raggio_nodo = settings.ray_dot
        self.spess_edge = settings.thickness_edge
        self.list_op_input = [self.build_local]

    def __call__(self, city, city2, pos, neig, p_sol):
        im = []
        for op in self.list_op_input:
            im.append(op(city, city2, pos, neig, p_sol))
        im = np.concatenate(im, axis=2)
        return normalize_image(im)

    def build_local(self, city, city2, pos, neig, p_sol):
        pos_go, max_value, p_go = transformation(pos, city, city2, neig)
        pixel_man = pixel_in_image(max_value=max_value,
                                   num_pixel=self.num_pixel, raggio=self.raggio_nodo)
        # red = (intensity, 0, 0)
        # green = (0, intensity, 0)
        # blue = (0, 0, intensity)
        im_red = np.zeros((self.num_pixel, self.num_pixel, 1), np.uint8)
        im_green = np.zeros((self.num_pixel, self.num_pixel, 1), np.uint8)
        im_blue = np.zeros((self.num_pixel, self.num_pixel, 1), np.uint8)

        # color the considered edge in the image
        c_x, c_y = pixel_man.pixel_pos(vec=pos_go[0])
        cv.circle(im_green, (c_x, c_y), self.raggio_nodo, intensity,
                  thickness=-1, lineType=cv.LINE_AA)

        t_x, t_y = pixel_man.pixel_pos(vec=pos_go[1])
        cv.circle(im_green, (t_x, t_y), self.raggio_nodo, intensity,
                  thickness=-1, lineType=cv.LINE_AA)

        im_green = cv.line(im_green, (c_x, c_y), (t_x, t_y), intensity, self.spess_edge,
                           lineType=cv.LINE_AA)

        for j in range(pos_go.shape[0]):
            c_x, c_y = pixel_man.pixel_pos(vec=pos_go[j])
            cv.circle(im_red, (c_x, c_y), self.raggio_nodo, intensity,
                      thickness=-1, lineType=cv.LINE_AA)

        for i, j in enumerate(p_go):
            for h in p_sol[j]:
                if h in neig:
                    c_x, c_y = pixel_man.pixel_pos(vec=pos_go[i])
                    cv.circle(im_blue, (c_x, c_y), self.raggio_nodo, intensity,
                              thickness=-1, lineType=cv.LINE_AA)
                    ind = np.argwhere(p_go == h)[0][0]
                    h_x, h_y = pixel_man.pixel_pos(vec=pos_go[ind])
                    cv.circle(im_blue, (h_x, h_y), self.raggio_nodo, intensity,
                              thickness=-1, lineType=cv.LINE_AA)
                    im_blue = cv.line(im_blue, (c_x, c_y), (h_x, h_y), intensity,
                                      self.spess_edge, lineType=cv.LINE_AA)
        return np.concatenate([im_blue, im_green, im_red], axis=-1)


class ImageTestCreator(ImageCreator):
    def __init__(self, settings, pos):
        super(ImageTestCreator, self).__init__(settings)
        self.dist_matrix = distance_mat(pos)
        pos -= np.mean(pos, axis=0)
        max_global = np.max(np.linalg.norm(pos, axis=1))
        self.pixel_global = pixel_in_image(max_value=max_global,
                                           num_pixel=settings.num_pixels,
                                           raggio=settings.ray_dot)
        self.pos = pos

        # preprocessing for candidates
        self.candidates_agent = CandidatesAgent(settings, self.dist_matrix)

    def get_image(self, city1, city2, p_sol):
        neig = self.candidates_agent.create_candidate(city1, city2)
        pos_go, max_value, p_go = transformation(self.pos, city1, city2, neig)
        pixel_man = pixel_in_image(max_value=max_value,
                                   num_pixel=self.num_pixel, raggio=self.raggio_nodo)

        distance_one = pixel_man.pixel_distance()
        too_close = True if self.dist_matrix[city1, city2] < 3 * distance_one else False

        red = (intensity, 0, 0)
        green = (0, intensity, 0)
        blue = (0, 0, intensity)
        im = np.zeros((self.num_pixel, self.num_pixel, 3), np.uint8)

        # color the considered edge in the image
        c_x, c_y = pixel_man.pixel_pos(vec=pos_go[0])
        cv.circle(im, (c_x, c_y), self.raggio_nodo, green,
                  thickness=-1, lineType=cv.LINE_AA)

        t_x, t_y = pixel_man.pixel_pos(vec=pos_go[1])
        cv.circle(im, (t_x, t_y), self.raggio_nodo, green,
                  thickness=-1, lineType=cv.LINE_AA)

        im = cv.line(im, (c_x, c_y), (t_x, t_y), green, self.spess_edge,
                     lineType=cv.LINE_AA)

        for j in range(pos_go.shape[0]):
            c_x, c_y = pixel_man.pixel_pos(vec=pos_go[j])
            cv.circle(im, (c_x, c_y), self.raggio_nodo, red,
                      thickness=-1, lineType=cv.LINE_AA)

        for i, j in enumerate(p_go):
            for h in p_sol[str(j)]:
                if h in neig:
                    c_x, c_y = pixel_man.pixel_pos(vec=pos_go[i])
                    cv.circle(im, (c_x, c_y), self.raggio_nodo, blue,
                              thickness=-1, lineType=cv.LINE_AA)
                    ind = np.argwhere(p_go == h)[0][0]
                    h_x, h_y = pixel_man.pixel_pos(vec=pos_go[ind])
                    cv.circle(im, (h_x, h_y), self.raggio_nodo, blue,
                              thickness=-1, lineType=cv.LINE_AA)
                    im = cv.line(im, (c_x, c_y), (h_x, h_y), blue, self.spess_edge,
                                 lineType=cv.LINE_AA)

        return normalize_image(im), too_close


class output_visual_checker:
    def __init__(self, settings, func):
        self.settings = settings
        self.num_pixel = settings.num_pixels
        self.raggio_nodo = settings.ray_dot
        self.spess_edge = settings.thickness_edge
        self.func = func

    def __call__(self, city, next_city, neig, pos, output_handler, p_sol):
        self.pos_new, self.max_value, pos_go = transformation(pos, city, next_city, neig)
        # self.pos_new = self.pos_new[self.settings.steps:]
        self.pixel_man = pixel_in_image(max_value=self.max_value,
                                        num_pixel=self.num_pixel, raggio=self.raggio_nodo)
        print(f"main vertex {city}")
        print(f"second extreme {next_city}")
        print(f"distance : {np.sqrt(np.sum((pos[city] - pos[next_city]) ** 2))}")
        print(f"neig {neig}")
        image, out = self.func(city, next_city, neig, pos, output_handler, p_sol)

        # list_images_to_plot = [image[:, :, 0], image[:, :, 1],
        #                        np.stack([image[:, :, 0], image[:, :, 1], image[:, :, 0]], axis=2)]
        #
        # plot_cv(list_images_to_plot, city)
        print(f"Is the edge optimal?:  {out}")
        plot_single_cv(image)
        print()
        return image, out
