import fitz  # PyMuPDF
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(pdf_file)
    text = ""

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    return text

def answer_question(question, text):
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer
