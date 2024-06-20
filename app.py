from flask import Flask, render_template, request
from pdf_utils import extract_text_from_pdf, answer_question

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# PDF upload and answer page
@app.route('/answer', methods=['POST'])
def answer():
    if 'pdf_file' not in request.files:
        return render_template('index.html', error="No PDF file uploaded.")

    pdf_file = request.files['pdf_file']

    if pdf_file.filename == '':
        return render_template('index.html', error="No PDF file selected.")

    pdf_chunks = extract_text_from_pdf(pdf_file)
    question = request.form['question']

    if not question:
        return render_template('index.html', error="Please enter a question.")

    answer = answer_question(question, pdf_chunks)
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
