import openai
import os
import pyaudio
import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

class VoiceAssistant():
    
    def __init__(self):
        self.microphone = sr.Microphone()
        self.recognizer = sr.Recognizer()
        self.player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    
    def detect_speech(self):
        while True:
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration = 0.5)
                    print("I'm listening...")
                    audio = self.recognizer.listen(source, timeout = 6.5)
                    text = self.recognizer.recognize_whisper_api(audio)
                    
                    return text
                
            except sr.RequestError as e:
                print(f"Could not request results from Whisper API: {e}")
                
            except sr.UnknownValueError:
                print("Unknown error occurred")
            
            except sr.WaitTimeoutError:
                print("No speech detected within the timeout period.")
                return False
    
    def get_completion(self, messages, model = "gpt-4-turbo-2024-04-09", temperature=0):
        response = openai.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens = 1500,
            temperature = temperature
        )
        
        message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        return message
    
    def speak(self, input_text):
        player_stream = self.player_stream
        
        with openai.audio.speech.with_streaming_response.create( 
            model="tts-1", 
            voice="onyx", 
            response_format="pcm", 
            input=input_text, 
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024): 
                player_stream.write(chunk)

# Main execution
v_assist = VoiceAssistant()
messages = [{'role': 'system', 'content': 'You are a voice assistant named J.A.R.V.I.S that responds in the style of the AI J.A.R.V.I.S from Iron Man. Respond in a friendly and helpful tone, with very concise answers.'}]

try:
    while True:
        is_speaking = v_assist.detect_speech()
        if is_speaking == "Stop." or "stop":
            break  # Exit the loop if the speaker says "stop"

        # Continue processing if speech is detected
        print("Processing speech...")
        print(is_speaking)
        messages.append({'role': 'user', 'content': is_speaking})
        response = v_assist.get_completion(messages, temperature = 0.5)
        v_assist.speak(response)
        print(response)
        
finally:
    print("Assistant stopped.")