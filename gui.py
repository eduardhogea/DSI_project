import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PyPDF2 import PdfReader
import docx
import threading
import requests

def chunk_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

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
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def classify_and_update_gui(text):
    def classification_task():
        base_url = "http://localhost:3000/"
        url = base_url + selected_model.get()
        candidate_labels = ["Art", "Science", "Technology", "Business", "Health"]  # Continue with your labels

        chunks = chunk_text(text, max_length=500)  # Adjust max_length as needed
        classification_counts = {}
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            data = {"text": chunk, "candidate_labels": candidate_labels}
            try:
                response = requests.post(url, json=data, timeout=100)
                response.raise_for_status()
                result = response.json()

                labels, scores = result
                max_label = labels[scores.index(max(scores))]

                classification_counts[max_label] = classification_counts.get(max_label, 0) + 1

                # Update progress bar
                progress = int((i+1) / total_chunks * 100)
                root.after(0, lambda progress=progress: loading_bar.__setitem__('value', progress))
                root.update_idletasks()
            except requests.exceptions.RequestException as e:
                print("An error has occurred while classifying the text:", e)

        final_classification = max(classification_counts, key=classification_counts.get)
        result_text = f"Final Classification: {final_classification}\n\nClassification Counts:\n"
        for label, count in classification_counts.items():
            result_text += f"{label}: {count}\n"

        def update_gui():
            result_display.config(state=tk.NORMAL)
            result_display.delete('1.0', tk.END)
            result_display.insert(tk.END, result_text)
            result_display.config(state=tk.DISABLED)
            loading_bar['value'] = 0  # Reset the progress bar

        root.after(0, update_gui)

    threading.Thread(target=classification_task).start()

def browse_files():
    filetypes = (('PDF files', '*.pdf'), ('Word files', '*.docx'), ('Text files', '*.txt'), ('All files', '*.*'))
    filename = filedialog.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
    if filename:
        process_file(filename)

root = TkinterDnD.Tk()
root.title('Drag and Drop File Converter')
root.geometry('600x800')

selected_model = tk.StringVar()
selected_model.set("classify_distilbert")

model_endpoints = ["classify_distilbert", "classify_mobilebert", "classify_roberta_large", "classify_squeezebert"]
model_dropdown = tk.OptionMenu(root, selected_model, *model_endpoints)
model_dropdown.pack(side=tk.TOP, pady=10)

label = tk.Label(root, text='Drag and drop a PDF, .docx, or .txt file here or browse to select')
label.pack(fill=tk.BOTH, expand=True)
label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', handle_file_drop)

loading_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
loading_bar.pack(side=tk.TOP, pady=10)

browse_button = tk.Button(root, text="Browse", command=browse_files, bg='blue', fg='white', padx=20, pady=10, font=('Helvetica', 12))
browse_button.pack(side=tk.BOTTOM, pady=10)

result_display = scrolledtext.ScrolledText(root, state=tk.DISABLED, height=10)
result_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()
