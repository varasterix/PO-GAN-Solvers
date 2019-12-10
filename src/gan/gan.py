import torch
import torch.nn.functional as F
from src.generator.generator import Generator
from src.discriminator.discriminator import Discriminator
from src.discriminator.checker import nb_cycles


class GAN:
    def __init__(self):
        self.input_shape = 10

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.loss_function = torch.nn.MSELoss()

    def train(self, epochs=1, batch_size=128):
        for epoch in range(epochs):
            input = torch.randn(1)
            output = self.generator(input)
            nb_cycle_pred = torch.tensor([nb_cycles(output.detach().numpy())])
            print(nb_cycle_pred)
            nb_cycle = torch.tensor([1])
            loss = self.loss_function(nb_cycle_pred, nb_cycle)

            self.generator.optimizer.zero_grad()
            loss.backward()
            self.generator.optimizer.step()


gan = GAN()
gan.train()
