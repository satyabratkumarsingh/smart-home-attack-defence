
from cyclegan_attack.cycle_gan_main import train_attack_model
from defence.defence_model import train_defence_model





def main():
    train_attack_model()
    train_defence_model()


if __name__ == "__main__":
    main()