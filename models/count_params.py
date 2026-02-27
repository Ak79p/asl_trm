import torch
from models.trm_micro import TRMMicro

def count_parameters(model):
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        print(f"{name:50s} | {n:8d} | trainable={param.requires_grad}")

    print("\n==============================")
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print("==============================\n")


def main():
    # num_classes = 300
    # model = BaselineTransformer(num_classes=num_classes)
    # model = TRMLite(num_classes=num_classes)
    model = TRMMicro(num_classes=2000)
    count_parameters(model)


if __name__ == "__main__":
    main()
