from torch.optim import Adam
import torch.nn as nn
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from sklearn.preprocessing import StandardScaler
from utils.file_utils import find_all_file_names
from generator.generator import Generator
from discriminator.discriminator import Discriminator
from dataset.cici_dataset import CICIDataset
from torch.utils.data import DataLoader
from model.train_model import train_model

DATASET_DIRECTORY = './../CICIoT2023/'
LEARNING_RATE = 0.0002
LAMBDA_ADVERS = 1
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_EPOCHS = 5
BATCH_SIZE = 1
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(): 
    csv_files = find_all_file_names(DATASET_DIRECTORY)
    input_features = len(pd.read_csv(csv_files[0]).columns) - 1

    gen_attack = Generator(input_features, input_features).to(DEVICE)
    gen_benign = Generator(input_features, input_features).to(DEVICE)
    des_attack = Discriminator(input_features, 10, 1).to(DEVICE)
    des_benign = Discriminator(input_features, 10, 1).to(DEVICE)

    opt_gen = Adam(list(gen_attack.parameters()) + list(gen_benign.parameters()), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(
        list(des_attack.parameters()) + list(des_benign.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    mse = nn.MSELoss()
    L1 = nn.L1Loss()
    discriminator_scaler = GradScaler()
    generator_scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):
        for csv_file in csv_files:
            data_frame = pd.read_csv(csv_file)
            data_frame_benign = data_frame.query('label == "BenignTraffic"')
            data_frame_attack = data_frame.query('label != "BenignTraffic"')

            attack_features_df = data_frame_attack.drop('label', axis=1)
            benign_features_df = data_frame_benign.drop('label', axis=1)

        
            # Initialize and fit the scaler
            attack_standard_scalar = StandardScaler()
            attack_scaled_data_features = attack_standard_scalar.fit_transform(attack_features_df)

            # Initialize and fit the scaler
            benign_standard_scalar = StandardScaler()
            benign_scaled_data_features = benign_standard_scalar.fit_transform(benign_features_df)

        
            dataset = CICIDataset(data_frame, attack_scaled_data_features, data_frame_attack['label'],
                                benign_scaled_data_features, data_frame_benign['label'])

            

            input_features = data_frame.shape[1] - 1
            gen_attack = Generator(input_features, input_features).to(DEVICE)
            gen_benign = Generator(input_features, input_features).to(DEVICE)
            des_attack = Discriminator(input_features, 10, 1).to(DEVICE)
            des_benign = Discriminator(input_features, 10, 1).to(DEVICE)


            loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2, pin_memory=True)
            train_model(loader,des_benign, des_attack, gen_benign, gen_attack, opt_disc, opt_gen,L1, mse,
                        discriminator_scaler, generator_scaler, attack_standard_scalar, benign_standard_scalar)
    #save_checkpoint()
if __name__ == "__main__":
    main()