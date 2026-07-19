# 🎧 HelpDesk AI Support Assistant

HelpDesk AI is a professional, intelligent customer support chatbot built with Python and Streamlit. It leverages dual AI engines to provide rapid problem resolution and features a modern, glassmorphism-inspired UI.

![Project Preview](preview.gif)

## ✨ Key Features

- **Dual AI Engines**: 
  - 🌐 **Cloud AI (GPT4Free)**: Advanced natural language understanding without requiring expensive API keys.
  - 🧠 **Local AI (PyTorch Seq2Seq)**: Fully offline, custom-trained AI built for specific company FAQs, guaranteeing data privacy.
- **⚡ Quick Actions**: One-click resolution for the most common user queries (e.g., Password Reset, Connection Errors).
- **📄 Transcript Export**: Allows users to download their entire chat session as a `.txt` file, simulating the creation of a real Customer Support Ticket.
- **💎 Premium UI**: Built with Streamlit, featuring a dark mode glassmorphism design, custom CSS, and a professional layout.

## 🚀 Installation & Usage

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **(Optional) Train the Local PyTorch Model:**
```bash
python train.py
```
*(This will train the offline Seq2Seq model using the english `data/faq.json` database).*

3. **Launch the Application:**
```bash
streamlit run app.py
```

## 🛠️ Technologies Used
- **Python 3**
- **Streamlit** (Frontend framework)
- **PyTorch** (Local neural network engine)
- **GPT4Free (g4f)** (Cloud AI integration)

---
*Created for the final jury presentation.*
