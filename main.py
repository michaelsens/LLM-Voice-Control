import speech_recognition as sr

def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()

    # Set up the microphone using sounddevice
    with sr.Microphone(device_index=1) as source:  # Adjust device_index if needed
        print("Please speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the speech.")
        return None
    except sr.RequestError:
        print("Could not request results from the speech recognition service.")
        return None

if __name__ == "__main__":
    text = recognize_speech_from_microphone()
