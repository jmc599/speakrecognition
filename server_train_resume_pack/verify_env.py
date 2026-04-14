import matplotlib
import torch
import torchaudio


def main():
    print("torch:", torch.__version__)
    print("torchaudio:", torchaudio.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
