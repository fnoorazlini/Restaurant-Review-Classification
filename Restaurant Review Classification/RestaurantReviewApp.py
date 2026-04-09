import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # For displaying images
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import re

class RestaurantReviewApp:
    def __init__(self, master):
        self.master = master
        master.title("Restaurant Review Classification App")
        master.geometry("800x600")
        master.config(bg="#F0F4F8")  # Light white-blue background

        # Load pre-trained model and vectorizer
        self.model = joblib.load('Restaurant_review_model')
        self.vectorizer = joblib.load('count_v_res')

        # Initialize frames
        self.home_frame = tk.Frame(master, bg="#F0F4F8")  # Light background
        self.main_frame = tk.Frame(master, bg="#F0F4F8")

        self.setup_home_frame()
        self.setup_main_frame()

        self.show_home()

    def setup_home_frame(self):
        # Welcome text
        logo = "🍽️ Welcome to Restaurant Review Classifier"
        logo_label = tk.Label(
            self.home_frame,
            text=logo,
            font=('Segoe UI', 28, 'bold'),
            bg="#F0F4F8",
            fg='#0D3B66'  # Dark blue
        )
        logo_label.pack(pady=20)

        welcome_message = (
            "Analyze your restaurant reviews with our intelligent classifier.\n"
            "Simply enter a review, and we'll determine whether it's positive or negative!"
        )
        welcome_label = tk.Label(
            self.home_frame,
            text=welcome_message,
            font=('Segoe UI', 16),
            bg="#F0F4F8",
            fg="#3A506B",  # Medium blue
            wraplength=600,
            justify="center"
        )
        welcome_label.pack(pady=20)

        # Add restaurant image below the welcome text
        try:
            self.restaurant_image = Image.open("image/restaurant.jpg")  # Use relative path to the image
            self.restaurant_image = self.restaurant_image.resize((300, 200), Image.Resampling.LANCZOS)  # Smaller size
            self.restaurant_photo = ImageTk.PhotoImage(self.restaurant_image)
            image_label = tk.Label(
                self.home_frame,
                image=self.restaurant_photo,
                bg="#F0F4F8"
            )
            image_label.pack(pady=20)
        except Exception as e:
            print(f"Error loading image: {e}")

        # Start button
        start_button = tk.Button(
            self.home_frame,
            text="Get Started",
            command=self.show_main,
            font=('Segoe UI', 16, 'bold'),
            bg="#00509E",  # Dark blue
            fg="#FFFFFF",  # White text
            activebackground="#0074D9",  # Medium blue on hover
            activeforeground="#F0F4F8",
            bd=0,
            padx=20,
            pady=10,
            relief="raised",
            cursor="hand2"
        )
        start_button.pack(pady=30)

    def setup_main_frame(self):
        # Add custom logo and title
        self.add_logo()

        title_font = ('Segoe UI', 18, 'bold')
        self.label = ttk.Label(
            self.main_frame,
            text="Enter your restaurant review:",
            font=title_font,
            anchor="center",
            background="#F0F4F8",
            foreground="#0D3B66"  # Dark blue
        )
        self.label.pack(pady=15)

        self.text_entry = tk.Text(
            self.main_frame,
            height=6,
            width=50,
            wrap="word",
            font=('Arial', 12),
            bg="#FFFFFF",
            bd=1,
            relief="solid",
            highlightthickness=0,
            highlightcolor="#A3CEF1",  # Light blue
            insertbackground='black'
        )
        self.text_entry.pack(pady=15)

        # Container frame for the buttons to align them horizontally
        button_frame = tk.Frame(self.main_frame, bg="#F0F4F8")
        button_frame.pack(pady=10)

        # Classify button
        self.classify_button = tk.Button(
            button_frame, 
            text="Classify", 
            command=self.classify_review, 
            font=('Segoe UI', 16, 'bold'),
            bg="#00509E",  # Dark blue
            fg="#FFFFFF",
            activebackground="#0074D9",  # Medium blue
            activeforeground="#F0F4F8",
            bd=0,
            padx=20,
            pady=10,
            relief="raised",
            cursor="hand2"
        )
        self.classify_button.pack(side="left", padx=10)

        # Reset button
        self.reset_button = tk.Button(
            button_frame, 
            text="Reset", 
            command=self.reset_fields, 
            font=('Segoe UI', 16, 'bold'),
            bg="#D62828",  # Red
            fg="#FFFFFF",
            activebackground="#F94144",  # Bright red on hover
            activeforeground="#F0F4F8",
            bd=0,
            padx=20,
            pady=10,
            relief="raised",
            cursor="hand2"
        )
        self.reset_button.pack(side="left", padx=10)

        # Progress bar for classifying process
        self.progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=15)

        # Result label
        self.result_label = ttk.Label(
            self.main_frame,
            text="Enter a review and press 'Classify' to get the sentiment.",
            font=('Segoe UI', 12),
            background="#F0F4F8",
            foreground="#3A506B"  # Medium blue
        )
        self.result_label.pack(pady=10)

        # Back to Home button
        self.home_button = tk.Button(
            self.main_frame, 
            text="Back to Home", 
            command=self.show_home, 
            font=('Segoe UI', 16, 'bold'),
            bg="#00509E",
            fg="#FFFFFF",
            activebackground="#0074D9",
            activeforeground="#F0F4F8",
            bd=0,
            padx=20,
            pady=10,
            relief="raised",
            cursor="hand2"
        )
        self.home_button.pack(pady=10)

    def add_logo(self):
        logo = "🍽️ Restaurant Review Classifier"
        logo_label = ttk.Label(
            self.main_frame, text=logo, font=('Segoe UI', 24, 'bold'), background="#F0F4F8", foreground='#0D3B66'
        )
        logo_label.pack(pady=20)

    def preprocess_text(self, text):
        ps = PorterStemmer()
        stop_words = set(stopwords.words("english"))

        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stop_words]
        review = " ".join(review)

        return review

    def classify_review(self):
        user_input = self.text_entry.get("1.0", "end-1c")
        if user_input:
            self.progress_bar["value"] = 0
            self.result_label.config(text="")
            self.master.update_idletasks()

            # Simulate progress
            for i in range(0, 101, 10):
                self.progress_bar["value"] = i
                self.master.update_idletasks()
                self.master.after(100)

            processed_input = self.preprocess_text(user_input)
            processed_input_vectorized = self.vectorizer.transform([processed_input])
            self.predict_sentiment(processed_input_vectorized)
        else:
            self.result_label.config(
                text="Please enter a review before clicking 'Classify'.",
                foreground="red",
                font=('Segoe UI', 14, 'italic')
            )

    def predict_sentiment(self, processed_input_vectorized):
        prediction = self.model.predict(processed_input_vectorized)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        self.result_label.config(text=f"Predicted Sentiment: {sentiment}", font=('Segoe UI', 18, 'bold'))

        # Display sentiment with different colors
        if sentiment == "Positive":
            self.result_label.config(foreground="#27AE60")  # Green for positive sentiment
        else:
            self.result_label.config(foreground="#E74C3C")  # Red for negative sentiment

    def reset_fields(self):
        self.text_entry.delete("1.0", "end")
        self.result_label.config(
            text="Enter a review and press 'Classify' to get the sentiment.",
            foreground="#3A506B",
            font=('Segoe UI', 12)
        )
        self.progress_bar["value"] = 0

    def show_home(self):
        self.main_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)

    def show_main(self):
        self.home_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = RestaurantReviewApp(root)
    root.mainloop()
