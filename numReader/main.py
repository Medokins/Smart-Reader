from data_preprocessor import getBoundingBoxes, readDigits
import cv2
import PySimpleGUI as sg

def create_gui():
    sg.theme('DarkGreen3')   

    layout = [  [sg.Column([[sg.Button('Exit' )]], element_justification='right', expand_x=True)],
            [sg.Text("Choose a file: ", pad=(0,10)), sg.Input(), sg.FileBrowse(key="-IN-")],
            [sg.Button("Submit", pad=(0,10))],
            [sg.Text("", key='-OUTPUT-', pad=(0,10))],
            [sg.Text("Image:", key='-TXT-', size=(25, 2), pad=(0, 0), expand_x=True, expand_y=False, visible=False),
            sg.Text("Numbers:", key='-TXT2-', size=(25, 2), pad=(0, 0), expand_x=True, expand_y=False, visible=False)],
            [sg.Image(key='-IMAGE-',  size=(256, 256), pad=(0,0), expand_x=False, expand_y=False),
            sg.Image(key='-IMAGE2-', size=(256, 256), pad=(0,0), expand_x=True, expand_y=False)]
            ]

    window = sg.Window('Reading numbers', layout, size=(800, 600), finalize=True)
    
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "Submit":
            # predict from given picture
            # val = model(values["-IN-"])

            # path, label = model_labels(val)
           # window['-OUTPUT-'].update(value = f"Model prediction: {label}")

            imgbytes = get_image(values['-IN-'])
            window['-TXT-'].update(visible=True)
            window['-TXT2-'].update(visible=True)
            #['-IMAGE-'].update(data = imgbytes)
            #imgbytes = get_image(path)
            #window['-IMAGE2-'].update(data = imgbytes)


window.close()


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
