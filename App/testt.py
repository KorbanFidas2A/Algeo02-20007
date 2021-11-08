from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    image = np.array(Image.open('./assets/pemandangan.jpg'))
    #image.tofile


if __name__ == "__main__":
    main()
    print("hello world")
