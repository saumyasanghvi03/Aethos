# Aethos - Medical AI Assistant

## Overview

Aethos is an advanced medical AI assistant powered by Google's Gemini Pro model, designed to provide intelligent healthcare information and support. Built with a modern tech stack, Aethos offers an interactive conversational interface for medical queries, health assessments, and general wellness guidance.

## Features

- **AI-Powered Medical Assistance**: Leverages Google's Gemini Pro 1.5 Flash model for accurate and contextual medical information
- **Conversational Interface**: Natural, chat-based interaction for easy communication
- **Comprehensive Health Support**: Answers medical questions, provides health tips, and offers general wellness guidance
- **Modern UI**: Built with Gradio for an intuitive and responsive user experience
- **Session Management**: Maintains conversation history for context-aware responses
- **Safety-First Design**: Includes appropriate disclaimers and encourages professional medical consultation

## Tech Stack

- **Python**: Core programming language
- **Gradio**: Web interface framework for creating interactive demos
- **Google Generative AI (Gemini Pro)**: Large language model for medical assistance
- **dotenv**: Environment variable management for secure API key handling

## Installation

### Prerequisites

- Python 3.7 or higher
- A Google API key for Gemini Pro access

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saumyasanghvi03/Aethos.git
   cd Aethos
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**:
   - Create a `.env` file in the root directory
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```
   - Alternatively, set the environment variable directly in your system

4. **Run the application**:
   ```bash
   python app.py
   ```

## Usage

1. Launch the application using `python app.py`
2. Open your web browser and navigate to the local URL displayed (typically `http://127.0.0.1:7860`)
3. Start chatting with Aethos by typing your medical questions or health concerns
4. Receive AI-powered responses based on the latest medical knowledge
5. Continue the conversation - Aethos maintains context throughout the session

### Example Queries

- "What are the symptoms of the flu?"
- "How can I improve my sleep quality?"
- "What should I know about managing stress?"
- "Can you explain what causes high blood pressure?"

## Important Disclaimer

⚠️ **Aethos is an AI assistant designed to provide general health information and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.**

## Project Structure

```
Aethos/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── style.css          # Custom styling
├── LICENSE            # Apache 2.0 License
└── README.md          # Project documentation
```

## Configuration

The application uses the following Gemini Pro model configuration:
- **Model**: gemini-1.5-flash
- **Temperature**: 0.7 (balanced creativity and consistency)
- **Top P**: 0.9 (nucleus sampling for response diversity)
- **Top K**: 40 (limits vocabulary for focused responses)
- **Max Output Tokens**: 2048 (comprehensive responses)

## Contributing

Contributions are welcome! If you'd like to improve Aethos:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact & Attribution

**Developer**: Saumya Sanghvi

**GitHub**: [@saumyasanghvi03](https://github.com/saumyasanghvi03)

**Repository**: [Aethos](https://github.com/saumyasanghvi03/Aethos)

---

### Acknowledgments

- Built with [Google Gemini Pro](https://deepmind.google/technologies/gemini/) for advanced AI capabilities
- Powered by [Gradio](https://www.gradio.app/) for the interactive interface
- Inspired by the vision of making healthcare information more accessible

---

*Made with ❤️ for better health accessibility*
