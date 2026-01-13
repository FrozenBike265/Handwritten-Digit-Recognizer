import ctypes
from tkinter import *
import win32gui
from PIL import ImageGrab
import numpy as np
import tensorflow as tf

ctypes.windll.shcore.SetProcessDpiAwareness(1)

# Incarcam modelul
model = tf.keras.models.load_model('mnist_adam+batch_150_epochs.h5')

def predict_digit(img):
    # Prelucram imaginea si spunem ce cifra este folosind modelul
    img = img.resize((28, 28))  # Resize imaginii pt ca MNIST este 28 x 28
    img = img.convert('L')  # Facem imaginea grayscale
    img = np.array(img)  # Convertim imaginea intr-un array
    img = img.reshape(1, 28, 28, 1)  # Ii dam reshape ca sa arate ca modelul
    img = (255 - img) / 255.0  # Normalizam pixelii -> ca sa fie in [0,1]
    res = model.predict([img])[0]  # Prezicem ce cifra este desenata
    return np.argmax(res), max(res)  # Returnam cifra si cat de sigur este modelul ca a ghicit corect

class App(Tk):
    def __init__(self):
        super().__init__()

        # Main Window
        self.title("Handwritten Digit Recognizer")  
        self.geometry("800x500")  
        self.resizable(False, False)  
        self.configure(bg="#f0f0f0") 

        self.x = self.y = 0 

        # ----------------------- Crearea obiectelor -----------------------

        # Locul unde desenam
        self.canvas = Canvas(
            self, width=400, height=400, bg="white", cursor="cross", relief=SOLID, bd=2
        )

        self.canvas.grid(row=0, column=0, padx=20, pady=(30, 10), columnspan=2)

        # Locul unde ne va arata rezultatul
        self.result_label = Label(
            self, text="Draw a digit!", font=("Helvetica", 20), bg="#f0f0f0", fg="#333"
        )
        
        self.result_label.grid(row=0, column=2, padx=10, pady=10, sticky=W)

        # Buton pentru a reseta desenul
        self.clear_button = Button(
            self, text="Clear Canvas", command=self.clear_canvas, bg="#f44336", fg="white", font=("Helvetica", 14), relief=RAISED
        )
        
        self.clear_button.grid(row=1, column=0, padx=20, pady=(10, 20), sticky=E)

        # Button pentru a vedea ce cifra spune modelul ca am desenat
        self.recognize_button = Button(
            self, text="Recognize", command=self.classify_handwriting, bg="#4CAF50", fg="white", font=("Helvetica", 14), relief=RAISED
        )
        
        self.recognize_button.grid(row=1, column=1, padx=20, pady=(10, 20), sticky=W)

        # ----------------------- Legaturile -----------------------

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_canvas(self):
        # Sterge canvas-ul si reseteaza raspunsul
        self.canvas.delete("all")  
        self.result_label.configure(text="Draw a digit!")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  
        rect = win32gui.GetWindowRect(HWND) 
        im = ImageGrab.grab(rect)
        # Prezicem ce cifra este desenata
        digit, acc = predict_digit(im) 
        # Afisam rezultatul si cat de siguri suntem
        self.result_label.configure(text=f"Digit: {digit}, Confidence: {int(acc * 100)}%") 

    # def classify_handwriting(self):
    #     # Corrected for DPI scaling using ctypes
    #     HWND = self.canvas.winfo_id()
    #     rect = win32gui.GetWindowRect(HWND)
    #     scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
    #     rect = (
    #         int(rect[0] * scale_factor),
    #         int(rect[1] * scale_factor),
    #         int(rect[2] * scale_factor),
    #         int(rect[3] * scale_factor),
    #     )
    #     im = ImageGrab.grab(bbox=rect)
        
    #     digit, acc = predict_digit(im)
    #     self.result_label.configure(text=f"Digit: {digit}, Confidence: {int(acc * 100)}%")
        

    def draw_lines(self, event):
        self.x = event.x  # Coordonata x a mouse-ului
        self.y = event.y  # Coordonata y a mouse-ului
        r = 8  # Raza cercului care se va forma atunci cand vom apasa click stanga
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill="black")

# Ruleaza aplicatia
if __name__ == "__main__":
    app = App() 
    app.mainloop() 
