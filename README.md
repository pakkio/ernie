# Ernie Chat Interface

A Gradio-based web interface for chatting with Baidu's Ernie 4.5 Turbo AI model.

## Features

- **Real-time streaming chat** with Ernie 4.5 Turbo
- **Multi-language support** with automatic language instruction
- **Cancel functionality** to stop ongoing requests
- **Conversation history** with context preservation
- **Clean response filtering** to remove thinking patterns

## Language Support

The interface supports responses in:
- Italian (default)
- English
- Spanish
- French
- German
- Portuguese
- Russian
- Chinese
- Japanese
- Korean

## Installation

1. Install dependencies:
```bash
poetry install
```

2. Run the application:
```bash
python test_ernie_api.py
```

## Usage

1. Select your preferred response language from the dropdown
2. Type your message in the text box
3. Click "Send" or press Enter to submit
4. Use "Cancel" to stop ongoing requests
5. Use "Clear" to reset conversation history

The interface will automatically add "Answer in [Language]" to your prompts based on your language selection.

## Requirements

- Python 3.10+
- gradio
- gradio-client

## License

This project is for educational and testing purposes.