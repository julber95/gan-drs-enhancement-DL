import torch
import os


# Fonction pour entraîner le discriminateur
def D_train(x, G, D, D_optimizer, criterion, device):
    # On commence par réinitialiser les gradients du discriminateur
    D.zero_grad()
    # Partie 1 : Calcul de la perte pour les données réelles
    x_real, y_real = x, torch.ones(x.shape[0], 1).to(device)  # Etiquettes : 1 pour les données réelles
    D_output = D(x_real)  # Passage des vraies données dans le discriminateur
    D_real_loss = criterion(D_output, y_real)  # Calcul de la perte pour les vraies données

    # Partie 2 : Calcul de la perte pour les données générées (fausses)
    z = torch.randn(x.shape[0], 100).to(device)  # Génération de bruit (vecteur latent de dimension 100)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)  # Génération de fausses données avec G
    D_output = D(x_fake)  # Passage des fausses données dans le discriminateur
    D_fake_loss = criterion(D_output, y_fake)  # Calcul de la perte pour les fausses données

    # Calcul de la perte totale du discriminateur
    D_loss = D_real_loss + D_fake_loss
    # Rétropropagation de la perte
    D_loss.backward()
    # Mise à jour des poids du discriminateur
    D_optimizer.step()

    # On retourne la valeur de la perte pour suivi
    return D_loss.item()


# Fonction pour entraîner le générateur
def G_train(x, G, D, G_optimizer, criterion, device):
    # On commence par réinitialiser les gradients du générateur
    G.zero_grad()

    # Génération de bruit pour créer de nouvelles données
    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)  # Etiquettes : 1 (objectif : tromper le discriminateur)

    G_output = G(z)  # Génération des données factices
    D_output = D(G_output)  # Passage des données factices dans le discriminateur
    G_loss = criterion(D_output, y)  # Calcul de la perte du générateur
    # Note : Le générateur "veut" que le discriminateur attribue 1 à ses données factices
    # Rétropropagation de la perte
    G_loss.backward()
    # Mise à jour des poids du générateur
    G_optimizer.step()
    # On retourne la valeur de la perte pour suivi
    return G_loss.item()


# Fonction pour sauvegarder les modèles entraînés
def save_models(G, D, folder):
    # On sauvegarde les poids des modèles dans des fichiers .pth
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))  # Générateur
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))  # Discriminateur


# Fonction pour charger un modèle de générateur sauvegardé
def load_model(G, folder):
    ckpt_path = os.path.join(folder, 'G.pth')  # Chemin vers le fichier du générateur
    # Chargement du fichier sauvegardé avec `map_location` pour compatibilité CPU/GPU
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    # Suppression éventuelle du préfixe 'module.' dans les noms des paramètres
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G


def dynamic_soft_truncation(z, D_scores):
    """
    Adjust soft truncation scale based on discriminator scores.
    """
    scaling_factors = D_scores / (1 - D_scores + 1e-6)
    scaling_factors = torch.clamp(scaling_factors, 0.5, 1.2)  # Limit extreme values
    return z * scaling_factors.unsqueeze(1)
