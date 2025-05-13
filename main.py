from training.train_ReachNet import train_model
if __name__ == "__main__":
    train_model("../g7_v100/sf", "../g7_v100/bup", "../g7_v100/ns", epochs=5)