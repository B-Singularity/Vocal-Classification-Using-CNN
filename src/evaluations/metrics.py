import torch

class Evaluation:
    def __init__(self, validation_loader, device='cuda'):
        self.validation_loader = validation_loader
        self.device = device

    def evalution_accuaracy(self, model):
        model.to(self.device)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

        accuracy = correct / total
        return accuracy
