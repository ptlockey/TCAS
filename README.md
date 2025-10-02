# TCAS Monte Carlo Streamlit App

This repository hosts a Streamlit application for exploring TCAS encounter scenarios. The maintained entry point for the app is [`calculator.py`](calculator.py).

## Getting started

1. Create a virtual environment (optional but recommended) and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run calculator.py
   ```

   Streamlit will print a local URL in the terminal. Open it in your browser to interact with the tool.

## Notes

* The legacy `Calculator` script has been removed because it was an incomplete copy of the app. Always launch the application via `calculator.py`.
* Streamlit runs in headless mode automatically when deployed to environments without a display.
