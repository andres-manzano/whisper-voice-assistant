# Whisper Voice Assistant

A voice assistant that leverages the power of the Speech-to-Text (STT) and Text-to-Speech (TTS) functionalities of the OpenAI Whisper technology, combined with OpenAI's language models and LangChain for intelligent, context-aware responses.

## Features

- Speech-to-Text (STT): Converts spoken language into text.
- Text-to-Speech (TTS): Converts text into natural-sounding speech.
- LangChain: Generates context-aware replies and maintains context across conversations for coherent interactions through conversation history
- Multi-Language Support: Supports multiple languages as provided by OpenAI's capabilities, making it accessible to a global audience.

## Prerequisites

- `OPENAI_API_KEY`

## Installation

1. **Create and activate a Virtual Environment:**

```sh
python -m venv venv
source venv\Scripts\activate
```
2. **Install dependencies:**

```sh
pip install -r requirements.txt
```

3. **Set Up Environment Variables:**

- Create a .env file in the project root directory.
- Add your OpenAI API key to the .env file:

```sh
OPENAI_API_KEY = your_openai_api_key
```

## Usage

1. **Run the Voice Assistant:**

```sh
python voice_assistant.py
```
Speak into your microphone, and the assistant will respond. The voice assistant should finish when you say only the word "Stop".

2. **Interact with the Voice Assistant:**

- Speak into your microphone.
- The assistant will process your speech, generate a response, and read it back to you.
- If you want to finish the Voice Assistant, you only have to say **"Stop"** and it will finish.