import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import gradcheck
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0002
LAMBDA_ADVERS = 1
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10

"""
 Train generator and descrimnator for attack and benign data  
"""

def train_model(loader, disc_benign, disc_attack, genr_benign, genr_attack,
    opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, attack_standard_scalar, benign_standard_scalar
):
    loop = tqdm(loader, leave=True)
    print(loop)
    for idx, (benign_feature, benign_label, attack_feature, attack_label) in enumerate(loop):
        benign = benign_feature.to(DEVICE)
        attack = attack_feature.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=False):
             # Calculate total discriminator loss for ATTACK
            fake_attack = genr_attack(benign)

            #print(fake_attack.item())

            disc_attack_real = disc_attack(attack)
            disc_attack_fake = disc_attack(fake_attack.detach())

        
            disc_attack_real_loss = mse(disc_attack_real, torch.ones_like(disc_attack_real))

            disc_attack_fake_loss = mse(disc_attack_fake, torch.zeros_like(disc_attack_fake))
            disc_attack_loss = disc_attack_real_loss + disc_attack_fake_loss

            # Calculate total discriminator loss for BENIGN
            fake_benign = genr_benign(attack)

            disc_benign_real = disc_benign(benign)
            disc_benign_fake = disc_benign(fake_benign.detach())

            disc_benign_real_loss = mse(disc_benign_real, torch.ones_like(disc_benign_real))
            disc_benign_fake_loss = mse(disc_benign_fake, torch.zeros_like(disc_benign_fake))

            disc_benign_loss = disc_benign_real_loss + disc_benign_fake_loss

            # Calculate total loss for descriminator
            disc_total_loss = (disc_attack_loss + disc_benign_loss) / 2

            disc_total_loss.backward()
            opt_disc.step()


        with torch.cuda.amp.autocast(enabled=False):
            # adversarial losses
            disc_attack_fake = disc_attack(fake_attack)
            disc_benign_fake = disc_benign(fake_benign)

            loss_gen_attack = mse(disc_attack_fake, torch.ones_like(disc_attack_fake))
            loss_gen_benign = mse(disc_benign_fake, torch.ones_like(disc_benign_fake))

            loss_advers = loss_gen_attack + loss_gen_benign

            cycle_attack = genr_attack(fake_attack)
            cycle_benign = genr_attack(fake_benign)
            cycle_attack_loss = l1(attack, cycle_attack)
            cycle_benign_loss = l1(benign, cycle_benign)

            loss_cycle= cycle_attack_loss + cycle_benign_loss

            generator_total_loss = (loss_advers * LAMBDA_ADVERS + loss_cycle * LAMBDA_CYCLE)

            generator_total_loss.backward()
            opt_gen.step()
 
        if idx % 200 == 0:
         
          # Convert tensor to a DataFrame
          benign_tensor_df = pd.DataFrame(fake_benign.detach().cpu().reshape(1, -1).numpy())
          print(fake_benign.shape)
          print('@@@@@@ BENIGN @@@@@@@@@')
          print(benign_standard_scalar.inverse_transform(benign_tensor_df))
          print('@@@@@@ ATTACK @@@@@@@@@')
          attack_tensor_df = pd.DataFrame(fake_attack.detach().cpu().reshape(1, -1).numpy())
          print(attack_standard_scalar.inverse_transform(attack_tensor_df))

          #print(attack_standard_scalar.inverse_transform(fake_attack.detach().cpu().reshape(-1, 1)))

        loop.set_postfix(descriminator_loss = disc_total_loss.item() / (idx +1),
                         generator_loss = generator_total_loss.item() / (idx +1))
    