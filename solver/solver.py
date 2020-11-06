import torch.nn as nn

from state_processing_module import StateProcessingModule, ImageProcessingModule, InstructionProcessingModule, MultimodalFusionModule
from state_processing_module import ImageProcessingModuleAlt
from policy_module import PolicyModule, PolicyModuleAlt


class Solver(nn.Module):
    def __init__(self, n_filters, vocab_size, emb_dim, policy_input_dim, hid_dim, n_actions):
        super().__init__()

        image_module = ImageProcessingModule(n_filters)
        instruction_module = InstructionProcessingModule(vocab_size, emb_dim)
        multimodal_fusion_module = MultimodalFusionModule(emb_dim, n_filters)
        self.state_processing_module = StateProcessingModule(image_module, instruction_module, multimodal_fusion_module)
        self.policy_module = PolicyModule(policy_input_dim, hid_dim, n_actions)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 2:
                nn.init.kaiming_normal_(p)

    def forward(self, image, instruction):
        fusion = self.state_processing_module(image, instruction)
        action, value = self.policy_module(fusion)
        return action, value


class SolverAlt(nn.Module):
    def __init__(self, n_filters, policy_input_dim, hid_dim, n_actions):
        super().__init__()

        self.image_module = ImageProcessingModuleAlt(n_filters)
        self.policy_module = PolicyModuleAlt(policy_input_dim, hid_dim, n_actions)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 2:
                nn.init.kaiming_normal_(p)

    def forward(self, image, instruction):
        image = self.image_module(image)
        action, value = self.policy_module(image)
        return action, value
