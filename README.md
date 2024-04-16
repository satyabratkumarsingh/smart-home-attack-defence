# smart-home-attack-defence

This project is divided into two main components:

1. cyclegan-attack: This part leverages the CycleGAN model to train a system capable of generating new attack patterns. By executing cycle_gan_main.py, the program will begin outputting newly simulated attack and benign data.
2. defense: This module is designed to classify the data as either attacks or benign. It utilizes a neural network for the training process.


Both sections of the project are trained using the comprehensive dataset from the Canadian Institute for Cybersecurity (CIC), which consists of 169 CSV files. These files are not included in the repository due to size limitations imposed by Git.