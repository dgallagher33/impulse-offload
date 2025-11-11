# Impulse Offloader

“If you have the impulse to talk about something, let’s capture that.”

Impulse Offloader is a local-first ADHD thought-capture tool. The project helps you quickly record thoughts as they arrive, organize them, and resurface insights later.

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure environment**
   Create a `.env` file (it is ignored by git) with:
   ```env
   OPENAI_API_KEY=your_key_here
   VAULT_PATH=/absolute/path/to/vault
   ```
3. **Capture your first impulse**
   ```bash
   python -m impulse_offloader.capture "My first captured thought."
   ```
4. **Process captured notes** *(optional MVP step)*
   ```bash
   python -m impulse_offloader.process
   ```

## Folder Structure

```
impulse-offloader/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── impulse_offloader/
│   ├── __init__.py
│   ├── capture.py
│   ├── process.py
│   ├── dashboard.py
│   └── utils/
│       ├── file_ops.py
│       ├── openai_helpers.py
│       └── whisper_integration.py
├── vault/
│   ├── Inbox/
│   └── Processed/
└── tests/
    ├── test_capture.py
    └── test_processing.py
```

## Milestones

| Version | Goal            | Description                   |
| ------- | --------------- | ----------------------------- |
| v0.1    | CLI capture     | Text input + file creation    |
| v0.2    | Voice capture   | Whisper transcription         |
| v0.3    | AI processing   | Summaries & tags              |
| v0.4    | Dashboard       | Daily resurfacing             |
| v1.0    | GUI / Desktop   | Streamlit or Tauri front-end  |

## Contributing

* Use PEP 8 / Black formatting.
* Add docstrings for every function.
* Prefer `pathlib.Path` for filesystem interactions.
* Avoid magic constants where possible.

## License

This project is licensed under the terms of the LICENSE file included in the repository.
