import os
import openai
import logging
import speech_recognition as sr
from pyaudio import PyAudio, paInt16
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain.schema.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load .env file that contains the OPENAI_API_KEY
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Set logging level to ERROR to hide warnings
logging.getLogger().setLevel(logging.ERROR)

class VoiceAssistant():
    
    def __init__(self):
        """
        Initializes the voice assistant.
        
        Attributes:
            self.microphone: A microphone instance to capture audio input.
            self.recognizer: A speech recognizer instance to process the captured audio.
            self.player_stream: An audio output stream configured with specific audio settings.
            self.chat_history_for_chain: An instance to store conversation history.
        """
        self.microphone = sr.Microphone()
        self.recognizer = sr.Recognizer()
        self.player_stream = PyAudio().open(format = paInt16, channels = 1, rate = 24000, output = True)
        self.chat_history_for_chain = ChatMessageHistory()
        
    def detect_speech(self):
        """
        Detects and recognizes speech using the Whisper API, continuously listening until speech is detected or an error occurs.
        
        Returns:
            str or bool: The recognized text if successful, or False if no speech is detected or an error occurs.
        """
        while True:
            try:
                with self.microphone as source:
                    # Adjust the recognizer sensitivity to ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration = 0.5)
                    # Signal to the user that the system is ready to listen.
                    print("I'm listening...")
                    # Listen the phrases and extract it into audio data.
                    audio = self.recognizer.listen(source, timeout = 6.5)
                    # Speech-to-Text using the Whisper API.
                    text = self.recognizer.recognize_whisper_api(audio)
                    
                    return text
                
            except sr.RequestError as e:
                # Handle errors from the Whisper API request.
                print(f"Could not request results from Whisper API: {e}")
                
            except sr.UnknownValueError:
                # Handle the case where speech was detected but not recognized.
                print("Unknown error occurred.")
            
            except sr.WaitTimeoutError:
                # Handle the case when no speech is detected within the timeout.
                print("No speech detected within the timeout period.")
                
                return False
    
    def chain(self, model = "gpt-4o", temperature = 0):
        """
        Creates and configures a conversational AI chain using the specified model and parameters.
        This method sets up a chat prompt template, initializes a language model with the given parameters, 
        and combines them into a Runnable chain that includes the handling of chat history. 
        The chain is used to generate responses based on the chat history and the current user prompt.
        
        Parameters:
            model (str): The model ID of the OpenAI GPT model to use for generating responses.
            temperature (float): The temperature setting for response generation, controlling the randomness. 
                                
        Returns:
            RunnableWithMessageHistory: A configured Runnable chain that processes user inputs, 
                                        generates responses using the chat history, and maintains the conversation context.
        """
        # System prompt providing instructions for the AI assistant.
        system_prompt = """
        You are a helpfull assistant using the chat history.
        Your answers should be friendly, helpful, concise,
        and include elements of wit and technical expertise.
        
        Adapt your tone based on the user's mood in the conversation.
        """
        
        # Define the chat prompt template with system instructions and placeholders for chat history and user input.
        prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    (
                        "human", "{prompt}"
                        ),
                ]
            )
        
        # Initialize the language model with the specified parameters.
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        
        # Combine the prompt template, language model, and output parser into a runnable chain.
        chain: Runnable = prompt_template | llm | StrOutputParser()
        
        # Configure the runnable chain to handle chat history, maintaining the context of the conversation.
        chain_with_message_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.chat_history_for_chain,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )
        
        return chain_with_message_history
    
    def tts(self, input_text):
        """
        Converts input text to spoken audio using OpenAI's TTS (text-to-speech) API and outputs it through an audio stream.
        
        Parameters:
            input_text (str): The text that needs to be converted to speech.
        
        The audio output is handled by PyAudio, which plays the speech through the default audio device (e.g., speakers).
        Each piece of the audio is processed in chunks to ensure smooth playback without requiring the entire audio clip to
        be loaded into memory at once.
        """
        # Reference to the player stream object for outputting audio
        player_stream = self.player_stream
        
        # Create a streaming response from OpenAI's TTS API configured with specific settings
        with openai.audio.speech.with_streaming_response.create( 
            model="tts-1", 
            voice="onyx", 
            response_format="pcm", 
            input=input_text, 
        ) as response:
            # Iterate over each chunk of audio data returned by the API and write it to the audio stream
            for chunk in response.iter_bytes(chunk_size=1024): 
                player_stream.write(chunk)