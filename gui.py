"""
GUI module for file processing and classification using Tkinter and BentoML.
Allows drag-and-drop and browsing for files, and displays classification results.
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinterdnd2 import DND_FILES, TkinterDnD
from PyPDF2 import PdfReader
import docx
import threading
import requests

def handle_file_drop(event):
    """ Handle file drop events by processing the dropped file. """
    process_file(event.data)

def process_file(file_path):
    """ Process the given file path based on its extension. """
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        text = extract_text_from_txt(file_path)
    else:
        text = "Unsupported file format"
    
    classify_and_update_gui(text)

def extract_text_from_pdf(file_path):
    """ Extract text from a PDF file. """
    text = ''
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    """ Extract text from a DOCX file. """
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file_path):
    """ Extract text from a TXT file. """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def classify_and_update_gui(text):
    """ Classify the text and update the GUI with the results. """
    def classification_task():
        base_url = "http://localhost:3000/"
        url = base_url + selected_model.get()  # Use the selected model endpoint
        candidate_labels = [
            "Art", "Science", "Technology", "Business", "Health",
            "Education", "Environment", "Politics", "Sports", "Entertainment",
            "Music", "Literature", "History", "Philosophy", "Religion",
            "Culture", "Travel", "Food", "Fashion", "Finance",
            "Law", "Social Issues", "Psychology", "Mathematics", "Engineering",
            "Biology", "Medicine", "Economics", "Film", "Marketing"
        ]
        data = {"text": text, "candidate_labels": candidate_labels}

        try:
            response = requests.post(url, json=data, timeout=100)
            response.raise_for_status()
            result = response.json()

            labels, scores = result
            scores = [float(score) for score in scores]  # Convert scores to float

            # Finding the label with the highest score
            max_score = max(scores)
            max_label = labels[scores.index(max_score)]

            # Snippet from the text (e.g., first 100 characters)
            text_snippet = text[:100] + '...' if len(text) > 100 else text

            # Construct the result text
            result_text = f"Classification:\n{text_snippet}\n\nTop Category: {max_label}, Score: {max_score:.4f}\n\nFull Classification:\n"
            for label, score in zip(labels, scores):
                result_text += f"Label: {label}, Score: {score:.4f}\n"
        except requests.exceptions.HTTPError as http_err:
            result_text = f"HTTP error occurred: {http_err}\nResponse content: {response.content}"
        except Exception as err:
            result_text = f"Other error occurred: {err}"

        def update_gui():
            result_display.config(state=tk.NORMAL)
            result_display.delete('1.0', tk.END)
            result_display.insert(tk.END, result_text)
            result_display.config(state=tk.DISABLED)

        root.after(0, update_gui)

    threading.Thread(target=classification_task).start()

def browse_files():
    """ Open a file dialog to browse files. """
    filetypes = (('PDF files', '*.pdf'), ('Word files', '*.docx'), ('Text files', '*.txt'), ('All files', '*.*'))
    filename = filedialog.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
    if filename:
        process_file(filename)

root = TkinterDnD.Tk()
root.title('Drag and Drop File Converter')
root.geometry('600x800')

selected_model = tk.StringVar()
selected_model.set("classify_distilbert")  # Default model

model_endpoints = [
    "classify_distilbert",
    "classify_mobilebert",
    "classify_roberta_large",
    "classify_squeezebert"
]

model_dropdown = tk.OptionMenu(root, selected_model, *model_endpoints)
model_dropdown.pack(side=tk.TOP, pady=10)

label = tk.Label(root, text='Drag and drop a PDF, .docx, or .txt file here or browse to select')
label.pack(fill=tk.BOTH, expand=True)
label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', handle_file_drop)

browse_button = tk.Button(root, text="Browse", command=browse_files, bg='blue', fg='white', padx=20, pady=10, font=('Helvetica', 12))
browse_button.pack(side=tk.BOTTOM, pady=10)

result_display = scrolledtext.ScrolledText(root, state=tk.DISABLED, height=10)
result_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()
