from training.train import train_model
if __name__ == "__main__":
    data_path="../g7_v100/sf/"
    train_model(data_path, epochs=10)