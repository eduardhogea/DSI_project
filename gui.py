import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog, scrolledtext
from PyPDF2 import PdfReader
import docx
import json
from zero_shot_transformer import ZeroShotClassifier
import threading

# Initialize the classifier
zero_shot_classifier = ZeroShotClassifier()

def handle_file_drop(event):
    process_file(event.data)

def process_file(file_path):
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        text = extract_text_from_txt(file_path)
    else:
        text = "Unsupported file format"
    
    save_text_as_json(text, 'output.json')
    classify_and_update_gui(text)

def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def save_text_as_json(text, output_file):
    with open(output_file, 'w') as f:
        json.dump({"content": text}, f, indent=4)

def browse_files():
    filetypes = (('PDF files', '*.pdf'), ('Word files', '*.docx'), ('Text files', '*.txt'), ('All files', '*.*'))
    filename = filedialog.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
    if filename:
        process_file(filename)

def classify_and_update_gui(text):
    def classification_task():
        candidate_labels = [
            "Art", "Science", "Technology", "Business", "Health",
            "Education", "Environment", "Politics", "Sports", "Entertainment",
            "Music", "Literature", "History", "Philosophy", "Religion",
            "Culture", "Travel", "Food", "Fashion", "Finance",
            "Law", "Social Issues", "Psychology", "Mathematics", "Engineering",
            "Biology", "Medicine", "Economics", "Film", "Marketing"
        ]

        result = zero_shot_classifier.classify(text, candidate_labels)
        result_text = f"\nClassification:\nLabel: {result['labels'][0]}, Score: {result['scores'][0]:.4f}"

        # Update the GUI with the result
        # Ensure the update is done in the main thread
        def update_gui():
            result_display.config(state=tk.NORMAL)
            result_display.delete('1.0', tk.END)
            result_display.insert(tk.END, result_text)
            result_display.config(state=tk.DISABLED)

        root.after(0, update_gui)

    # Start the classification in a new thread
    threading.Thread(target=classification_task).start()


root = TkinterDnD.Tk()
root.title('Drag and Drop File Converter')
root.geometry('600x800')

label = tk.Label(root, text='Drag and drop a PDF, .docx, or .txt file here or browse to select')
label.pack(fill=tk.BOTH, expand=True)
label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', handle_file_drop)

browse_button = tk.Button(root, text="Browse", command=browse_files, bg='blue', fg='white', padx=20, pady=10, font=('Helvetica', 12))
browse_button.pack(side=tk.BOTTOM, pady=10)

result_display = tk.scrolledtext.ScrolledText(root, state=tk.DISABLED, height=10)
result_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()