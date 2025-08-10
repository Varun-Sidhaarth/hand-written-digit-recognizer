import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import io

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        
        # Load the trained model
        try:
            self.model = tf.keras.models.load_model('digit_recognizer_model.h5')
        except:
            print("Error: Model file 'digit_recognizer_model.h5' not found!")
            print("Please run digit_recognizer.py first to train the model.")
            root.destroy()
            return

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create canvas for drawing
        self.canvas = tk.Canvas(self.main_frame, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # Create a PIL image for drawing
        self.image = Image.new('RGB', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)  # Left click
        self.canvas.bind('<B1-Motion>', self.draw_line)  # Left click drag
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)  # Left click release
        self.canvas.bind('<Button-3>', self.clear_canvas)  # Right click to clear
        
        # Create buttons
        self.clear_button = ttk.Button(self.main_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.predict_button = ttk.Button(self.main_frame, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=1, column=1, padx=5, pady=5)
        
        # Create prediction label
        self.prediction_label = ttk.Label(self.main_frame, text="Draw a digit (0-9)", font=('Arial', 14))
        self.prediction_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Initialize drawing variables
        self.old_x = None
        self.old_y = None
        self.line_width = 20
        self.line_color = 'black'
        self.is_drawing = False

    def start_draw(self, event):
        self.is_drawing = True
        self.old_x = event.x
        self.old_y = event.y

    def draw_line(self, event):
        if not self.is_drawing:
            return
            
        if self.old_x and self.old_y:
            # Draw on canvas
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                  width=self.line_width, fill=self.line_color,
                                  capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
            # Draw on PIL image
            self.draw.line([self.old_x, self.old_y, event.x, event.y],
                         fill='black', width=self.line_width)
        self.old_x = event.x
        self.old_y = event.y

    def stop_draw(self, event):
        self.is_drawing = False
        self.old_x = None
        self.old_y = None
        # Add a small delay before prediction to ensure the drawing is complete
        self.root.after(100, self.predict_digit)

    def clear_canvas(self, event=None):
        self.canvas.delete("all")
        self.image = Image.new('RGB', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit (0-9)")
        self.is_drawing = False
        self.old_x = None
        self.old_y = None

    def predict_digit(self, event=None):
        # Convert PIL image to grayscale and resize
        img = self.image.convert('L')
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert colors
        img_array = img_array.astype('float32') / 255.0
        
        # Add padding to center the digit
        img_array = self.center_digit(img_array)
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28)
        
        # Get prediction
        prediction = self.model.predict(img_array)
        predicted_digit = np.argmax(prediction[0])
        confidence = prediction[0][predicted_digit] * 100
        
        # Update prediction label
        self.prediction_label.config(
            text=f"Predicted: {predicted_digit}\nConfidence: {confidence:.2f}%"
        )

    def center_digit(self, img_array):
        # Find the bounding box of the digit
        rows = np.any(img_array > 0.1, axis=1)
        cols = np.any(img_array > 0.1, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Calculate the center of the digit
        rcenter = (rmin + rmax) // 2
        ccenter = (cmin + cmax) // 2
        
        # Calculate the shift needed to center the digit
        rshift = 14 - rcenter
        cshift = 14 - ccenter
        
        # Create a new array with the digit centered
        centered = np.zeros((28, 28))
        for r in range(28):
            for c in range(28):
                if 0 <= r + rshift < 28 and 0 <= c + cshift < 28:
                    centered[r, c] = img_array[r + rshift, c + cshift]
        
        return centered

def main():
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 