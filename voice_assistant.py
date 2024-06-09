from utils.assistant import VoiceAssistant

# Instance of the VoiceAssistant class.
v_assist = VoiceAssistant()

try:
    # Continuously process input speech until a stop command is detected.
    while True:
        # Detect speech using the voice assistant's method. It listens and returns the text spoken.
        speech = v_assist.detect_speech()
        # Check if the detected speech is a command to stop the assistant.
        if speech == "Stop.":
            break   # Break the loop and end the process if "Stop." is spoken.
        
        # Output to the console that the system is processing the detected speech.
        print("Processing speech...")
        # Display the recognized speech text.
        print(speech)
        
        # Generate a response based on the conversation history.
        chain_with_message_history = v_assist.chain(temperature = 0.5)
        response = chain_with_message_history.invoke(
            {"prompt": speech},
            config={"configurable": {"session_id": "unused"}},
        )
        # Speak out the generated response.
        v_assist.tts(response)
        # Output the response to the console for monitoring.
        print(response)
        
finally:
    # Ensure that a final message is printed when the assistant stops running.
    print("Assistant stopped.")