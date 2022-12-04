from data_preprocessor import getBoundingBoxes, readDigits

if __name__ == '__main__':
    name = 'test'
    coordinates_array = getBoundingBoxes(name, visualize=False)
    readDigits(coordinates_array, name)