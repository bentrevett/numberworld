import torch.nn as nn
import torch.nn.functional as F


class PolicyModule(nn.Module):
    def __init__(self, input_dim, hid_dim, n_actions):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)
        self.fc_a = nn.Linear(hid_dim, n_actions)
        self.fc_v = nn.Linear(hid_dim, 1)

    def forward(self, fusion):
        # fusion = [batch size, n_filters, height*, width*]
        batch_size, *_ = fusion.shape
        fusion = fusion.view(batch_size, -1)
        hidden = F.relu(self.fc_1(fusion))
        hidden = F.relu(self.fc_2(hidden))
        # hidden = [batch size, hid dim]
        action = self.fc_a(hidden)
        # action = [batch size, n_actions]
        value = self.fc_v(hidden)
        # value = [batch size, 1]
        return action, value


class PolicyModuleAlt(nn.Module):
    def __init__(self, input_dim, hid_dim, n_actions):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)
        self.fc_a = nn.Linear(hid_dim, n_actions)
        self.fc_v = nn.Linear(hid_dim, 1)

    def forward(self, image):
        # fusion = [batch size, n_filters, height*, width*]
        batch_size, *_ = image.shape
        image = image.reshape(batch_size, -1)
        hidden = F.relu(self.fc_1(image))
        hidden = F.relu(self.fc_2(hidden))
        # hidden = [batch size, hid dim]
        action = self.fc_a(hidden)
        # action = [batch size, n_actions]
        value = self.fc_v(hidden)
        # value = [batch size, 1]
        return action, value
