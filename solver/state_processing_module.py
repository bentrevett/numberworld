import torch
import torch.nn as nn
import torch.nn.functional as F


class StateProcessingModule(nn.Module):
    def __init__(self, image_module, instruction_module, multimodal_module):
        super().__init__()
        self.image_module = image_module
        self.instruction_module = instruction_module
        self.multimodal_module = multimodal_module

    def forward(self, image, instruction):
        image = self.image_module(image)
        instruction = self.instruction_module(instruction)
        fusion = self.multimodal_module(image, instruction)
        return fusion


class ImageProcessingModule(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=n_filters,
                               kernel_size=7,
                               stride=7)

    def forward(self, observation):
        # observation = [batch size, 3, height, width]
        observation = F.relu(self.conv1(observation))
        # observation = [batch size, n_filters, height*, width*]
        # height and width depend on world size, kernel size, stride, etc.
        return observation


class ImageProcessingModuleAlt(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=n_filters*2,
                               kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels=n_filters*2,
                               out_channels=n_filters,
                               kernel_size=7)
        self.conv3 = nn.Conv2d(in_channels=n_filters,
                               out_channels=n_filters//2,
                               kernel_size=7)

    def forward(self, observation):
        # observation = [batch size, 3, height, width]
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        # observation = [batch size, n_filters, height*, width*]
        # height and width depend on world size, kernel size, stride, etc.
        return observation


class InstructionProcessingModule(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, instruction):
        # instruction = [batch size, 2]
        instruction = self.embedding(instruction)
        # instruction = [batch size, 2, emb dim]
        instruction = instruction.mean(dim=1)
        # instruction = [batch size, emb dim]
        return instruction


class MultimodalFusionModule(nn.Module):
    def __init__(self, emb_dim, n_filters):
        super().__init__()

        self.fc_h = nn.Linear(emb_dim, n_filters)

    def forward(self, image, instruction):
        # image = [batch size, n_filters, height*, width*]
        # instruction = [batch size, emb dim]
        batch_size, n_filters, height, width = image.shape
        a = torch.sigmoid(self.fc_h(instruction))
        # a = [batch size, n_filters]
        m = a.unsqueeze(-1).unsqueeze(-1)
        # m = [batch size, n_filters, height*, width*]
        out = (m * image).contiguous()
        # out = [batch size, n_filters, height*, width*]
        return out
