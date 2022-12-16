from data_preprocessor import getBoundingBoxes, readDigits
import cv2
import PySimpleGUI as sg

def create_gui():
    sg.theme('DarkGreen3')   

    layout = [
            [sg.Column([[sg.Button('Exit' )]], element_justification='right', expand_x=True)],
            [sg.Text("Choose a file: ", pad=(0,10)), sg.Input(), sg.FileBrowse(key="-IN-")],
            [sg.Button("Submit", pad=(0,10))], [sg.Button("Live View", pad=(0,10))],
            [sg.Text("", key='-OUTPUT-', pad=(0,10))],
            [sg.Text("Image:", key='-TXT-', size=(25, 2), pad=(0, 0), expand_x=True, expand_y=False, visible=False), sg.Text("Numbers:", key='-TXT2-', size=(25, 2), pad=(0, 0), expand_x=True, expand_y=False, visible=False)],
            [sg.Image(key='-IMAGE-',  size=(256, 256), pad=(0,0), expand_x=False, expand_y=False), sg.Image(key='-IMAGE2-', size=(256, 256), pad=(0,0), expand_x=True, expand_y=False)]
            ]

    window = sg.Window('Reading numbers', layout, size=(800, 600), finalize=True)
    last_frame = None
    event = None

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "Submit":
            name = 'test'
            img = cv2.imread(f'../handwrittenNumbers/{name}.png')
            coordinates_array = getBoundingBoxes(img)
            readDigits(coordinates_array, img)
            break

        if event == "Live View":
            break

    window.close()

    if event == "Live View":
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            _, frame = vid.read()
            last_frame = frame.copy()
            getBoundingBoxes(img = frame, visualize=True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        vid.release()
        cv2.destroyAllWindows()
        coordinates_array = getBoundingBoxes(img=last_frame, visualize=False)
        last_frame_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        readDigits(coordinates_array, img=last_frame_gray)

if __name__ == '__main__':
    create_gui()