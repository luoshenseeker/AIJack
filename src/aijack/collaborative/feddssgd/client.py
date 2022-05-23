import copy

from ..core import BaseClient
import torch
import numpy as np


class FedDSSGDClient(BaseClient):
    def __init__(self, model, user_id=0, lr=0.1, upload_p = 0.1, send_gradient=True, device="cpu"):
        super(FedDSSGDClient, self).__init__(model, user_id=user_id)
        self.lr = lr
        self.send_gradient = send_gradient
        self.upload_p = upload_p
        self.device = device

        self.prev_parameters = []
        for param in self.model.parameters():
            self.prev_parameters.append(copy.deepcopy(param))

    def upload(self):
        if self.send_gradient:
            return self.upload_gradients()
        else:
            return self.upload_parameters()

    def upload_parameters(self):
        return self.model.state_dict()

    def upload_gradients(self):
        gradients = []
        for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
            gradients.append((prev_param - param) / self.lr)
        if self.upload_p < 1:
            gradients_numpy = [i.cpu().detach().numpy() for i in gradients]
            gradients_vector_one_dimention = [i.reshape(-1) for i in gradients_numpy]
            gradients_vector_one_dimention = np.concatenate(gradients_vector_one_dimention)
            gradients_vector_one_dimention = abs(gradients_vector_one_dimention)
            gradients_vector_one_dimention.sort()
            delete_capacity = round(gradients_vector_one_dimention.size * self.upload_p)
            threashold = gradients_vector_one_dimention[delete_capacity]
            gradients_numpy_clip = [np.clip(i, -threashold, threashold) for i in gradients_numpy]
            handled_gradient = []
            deleted_num = 0
            for clip, origin in zip(gradients_numpy_clip, gradients_numpy):
                zero_place = clip != origin
                clip[zero_place] = 0
                deleted_num += (zero_place.size - zero_place.sum())
                origin = origin - clip
                handled_gradient.append(origin)
            gradients = [torch.from_numpy(i).to(self.device) for i in handled_gradient]
            delete_persentage = deleted_num / gradients_vector_one_dimention.size
        return gradients

    def download(self, model_parameters):
        self.model.load_state_dict(model_parameters)

        self.prev_parameters = []
        for param in self.model.parameters():
            self.prev_parameters.append(copy.deepcopy(param))
