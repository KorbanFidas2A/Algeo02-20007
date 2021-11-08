from os import name
from PIL import Image
import numpy as np

def main():
    image = np.array(Image.open('../pemandangan.jpg'))


if __name__ == "__main__":
    main()