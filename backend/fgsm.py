import torch
import torch.nn as nn # FIXED: Import nn for the loss function

class Attack:
    def __init__(self, model):
        self.model = model

    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def run_attack(self, image_tensor, label, epsilon):
        image_tensor.requires_grad = True
        output = self.model(image_tensor)
        
        # Use CrossEntropyLoss for models that output raw logits
        loss = nn.CrossEntropyLoss()(output, label)

        self.model.zero_grad()
        loss.backward()
        image_grad = image_tensor.grad.data
        perturbed_image = self.fgsm_attack(image_tensor, epsilon, image_grad)
        return perturbed_image