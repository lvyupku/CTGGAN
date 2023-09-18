import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, csv_file):
        data_frame = pd.read_csv(csv_file, header=None, nrows=1600)
        self.data_frame = self._normalize(data_frame.iloc[:, :1000])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return torch.tensor(self.data_frame.iloc[idx].values).float()  

    def _normalize(self, df):
        return df / 238.5  # Normalize the data to the range [0, 1]


# Modified Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            nn.ConvTranspose1d(1, 1024, kernel_size=1, stride=1, padding=0),

            nn.ConvTranspose1d(1024, 512, kernel_size=26, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.ConvTranspose1d(512, 128, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, output_dim, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Upsample(1000, mode='linear', align_corners=False)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        return self.gen(x)


# Modified Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(128, 512, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(512, 1024, kernel_size=26, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)



# Load the data
data_path = "moving_data/预处理过的数据/data.csv"
dataset = MyDataset(data_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the models
input_dim = 100
output_dim = 1

# Instantiate the models and move them to the GPU
generator = Generator(input_dim, output_dim).to(device)
discriminator = Discriminator(output_dim).to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

num_epochs = 300
global_step = 0


def save_fake_data(epoch, fake_data):
    fake_data_2d = fake_data.view(fake_data.shape[0], -1)
    # Save the generated data as a CSV file
    np.savetxt(f"gen_data/fake_data_{epoch}.csv", fake_data_2d.detach().cpu().numpy(), delimiter=",")

    labels = np.zeros((fake_data.shape[0], 1))
    np.savetxt(f"gen_data/fake_labels_{epoch}.csv", labels, delimiter=",")
    print(f"Saved generated data to fake_data_{epoch}.csv")

# Training loop
for epoch in range(num_epochs):
    for i, real_data in enumerate(dataloader):
        real_data = real_data.unsqueeze(1).to(device)
        batch_size = real_data.shape[0]

        # Train the discriminator with 5 updates
        for _ in range(5):
            discriminator_optimizer.zero_grad()

            # Real data
            real_output = discriminator(real_data)
            real_labels = torch.ones(*real_output.shape).to(device)
            real_loss = criterion(real_output, real_labels)

            # Fake data
            noise = torch.randn(batch_size, input_dim, 1).to(device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())
            fake_labels = torch.zeros(*fake_output.shape).to(device)
            fake_loss = criterion(fake_output, fake_labels)
            
            # Gradient penalty
            epsilon = torch.rand(batch_size, 1, 1).to(device)
            x_hat = epsilon * real_data + (1 - epsilon) * fake_data.detach()
            x_hat.requires_grad_(True)
            hat_output = discriminator(x_hat)
            gradients = torch.autograd.grad(outputs=hat_output, inputs=x_hat,
                                            grad_outputs=torch.ones(hat_output.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(batch_size, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            lambda_gp = 10
            discriminator_loss = real_loss + fake_loss + lambda_gp * gradient_penalty
            
            # Backpropagation
            discriminator_loss.backward()
            discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()
        output = discriminator(fake_data)
        generator_real_labels = torch.ones(*output.shape).to(device)
        generator_loss = criterion(output, generator_real_labels)
        
        # Backpropagation
        generator_loss.backward()
        generator_optimizer.step()

        if i == len(dataloader) - 1:  # Check if it is the last iteration
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], D_loss: {discriminator_loss.item()}, G_loss: {generator_loss.item()}")
            if epoch % 100 == 0:
                save_fake_data(epoch, fake_data)

# Save the model parameters
torch.save(generator.state_dict(), 'generator.ckpt')
torch.save(discriminator.state_dict(), 'discriminator.ckpt')
