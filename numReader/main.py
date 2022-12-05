from data_preprocessor import getBoundingBoxes, readDigits
import cv2

if __name__ == '__main__':
    name = 'test'
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = vid.read()
        coordinates_array = getBoundingBoxes('CamerView', visualize=True, img=frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
    readDigits(coordinates_array, name)