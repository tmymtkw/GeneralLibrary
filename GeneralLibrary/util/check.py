import torch.cuda as cuda

def main():
    if cuda.is_available():
        print("INFO: CUDAは使用可能")
    else:
        print("WARNING : CUDAは使用不可")

if __name__ == "__main__":
    main()