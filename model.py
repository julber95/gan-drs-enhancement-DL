import torch
import torch.nn as nn  # Contient les blocs de base pour construire les modèles
import torch.nn.functional as F  # Contient des fonctions d'activation comme LeakyReLU



class Generator(nn.Module):  # Hérite de nn.Module pour définir un modèle PyTorch
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()  # Initialise la classe parente       
        self.fc1 = nn.Linear(100, 256)  # Couche 1 : réduction de la dimension du bruit (vecteur latent)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)  # Couche 2 : augmenter les dimensions
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)  # Couche 3 : encore plus de dimensions
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)  # Couche 4 : sortie à la dimension de l'image

    # Méthode forward : décrit le passage des données à travers le générateur
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)  # Leaky ReLU avec un alpha de 0.2 pour éviter les gradients nuls
        x = F.leaky_relu(self.fc2(x), 0.2)  # Activation non-linéaire pour chaque couche cachée
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))  # Sortie : valeurs entre -1 et 1 grâce à tanh


class Discriminator(nn.Module):  # Hérite également de nn.Module
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)  # Couche 1 : augmenter la dimension initiale
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)  # Couche 2 : réduire progressivement
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)  # Couche 3 : idem
        self.fc4 = nn.Linear(self.fc3.out_features, 1)  # Couche 4 : sortie binaire (réel ou faux)

    # Méthode forward : décrit le passage des données à travers le discriminateur
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)  # Activation LeakyReLU
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))  # Sortie : probabilité (0=faux, 1=réel)
