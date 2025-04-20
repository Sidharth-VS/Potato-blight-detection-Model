import torch
import torchvision.transforms as transforms
import numpy as np

class Model():
    def __init__(self):
        self.model = torch.jit.load("./model/Blight_model_scripted.pt")
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
        self.classes = ["Early Blight", "Late Blight", "Healthy"]

    def preprocess(self, img):
        img_tensor = self.transform(img)
        input = img_tensor.unsqueeze(0)
        return input

    def getPrediction(self, img):
        input = self.preprocess(img)
        output = self.model(input)
        idx = torch.argmax(output)
        return self.classes[idx]
    



     
