import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import customtkinter as ctk
import speech_recognition as sr
import pyttsx3
import threading
import os

# --- Configuration ---
DATASET_FILE = "dataset mecánica cuántica.csv"
SIMILARITY_THRESHOLD = 0.2

# --- Chatbot Logic ---
class QuantumBot:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        # Combine title and text for better matching
        self.df['combined_text'] = self.df['titulo'] + " " + self.df['texto']
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'].fillna(''))

    def get_response(self, user_input):
        user_tfidf = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        best_match_index = similarities.argmax()
        best_score = similarities[best_match_index]

        if best_score > SIMILARITY_THRESHOLD:
            return self.df.iloc[best_match_index]['texto']
        else:
            return "Lo siento, mi conocimiento se limita a la mecánica cuántica y no encuentro una respuesta en mi base de datos para esa pregunta."

# --- GUI Application ---
class ChatApp(ctk.CTk):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.title("Chatbot Mecánica Cuántica")
        self.geometry("600x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Voice Engine
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()

        self._setup_ui()

    def _setup_ui(self):
        # Chat History Area
        self.chat_frame = ctk.CTkScrollableFrame(self, width=580, height=550)
        self.chat_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Input Area
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=10, padx=10, fill="x")

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Escribe tu pregunta aquí...")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", lambda event: self.send_message())

        self.send_btn = ctk.CTkButton(self.input_frame, text="Enviar", width=80, command=self.send_message)
        self.send_btn.pack(side="right", padx=(0, 5))

        self.voice_btn = ctk.CTkButton(self.input_frame, text="🎤", width=40, command=self.start_voice_input)
        self.voice_btn.pack(side="right")

    def display_message(self, message, sender="Bot"):
        msg_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        msg_frame.pack(fill="x", pady=5)

        if sender == "User":
            lbl = ctk.CTkLabel(msg_frame, text=f"Tú: {message}", anchor="e", justify="right", text_color="#4da6ff", wraplength=500)
            lbl.pack(side="right", padx=10)
        else:
            lbl = ctk.CTkLabel(msg_frame, text=f"Bot: {message}", anchor="w", justify="left", text_color="#ffffff", wraplength=500)
            lbl.pack(side="left", padx=10)
            
            # Speak response in a separate thread to not freeze UI
            threading.Thread(target=self.speak, args=(message,), daemon=True).start()

        # Auto scroll to bottom
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def send_message(self):
        user_text = self.entry.get()
        if not user_text.strip():
            return

        self.display_message(user_text, "User")
        self.entry.delete(0, "end")

        # Get response
        response = self.bot.get_response(user_text)
        self.display_message(response, "Bot")

    def start_voice_input(self):
        self.voice_btn.configure(state="disabled", text="...")
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        try:
            # Check if PyAudio is available by trying to create a Microphone instance
            try:
                source = sr.Microphone()
            except OSError:
                self.after(0, lambda: self.display_message("Error: No se encontró un micrófono.", "Bot"))
                return
            except AttributeError:
                 self.after(0, lambda: self.display_message("Error: PyAudio no está instalado. El reconocimiento de voz no funcionará.", "Bot"))
                 return
            
            with source:
                try:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source)
                    print("Escuchando...")
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio, language="es-ES")
                    
                    # Update UI from main thread
                    self.after(0, lambda: self._process_voice_input(text))
                except sr.UnknownValueError:
                    pass # No speech detected
                except sr.RequestError:
                    self.after(0, lambda: self.display_message("Error de conexión para reconocimiento de voz.", "Bot"))
                except Exception as e:
                    print(f"Error: {e}")
        except Exception as e:
             self.after(0, lambda: self.display_message(f"Error crítico de voz: {str(e)}", "Bot"))
        finally:
            self.after(0, lambda: self.voice_btn.configure(state="normal", text="🎤"))

    def _process_voice_input(self, text):
        self.display_message(text, "User")
        response = self.bot.get_response(text)
        self.display_message(response, "Bot")

    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass # Handle potential audio driver issues gracefully

if __name__ == "__main__":
    if not os.path.exists(DATASET_FILE):
        print(f"Error: No se encontró el archivo {DATASET_FILE}")
    else:
        bot = QuantumBot(DATASET_FILE)
        app = ChatApp(bot)
        app.mainloop()
