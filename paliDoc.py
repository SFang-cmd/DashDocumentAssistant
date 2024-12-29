from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
import os
import torch
import base64

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

class paliDoc:
   
  def __init__(self, directory="./", list_ind=None, model="GPT"):
      if (list_ind == None):
        self.RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
      else:
        for ind in list_ind:
          self.RAG = RAGMultiModalModel.from_index(ind)
      self.directory = directory
      self.images = []
      self.num_pages = {}
      self.doc_num = 0
      if model == "GPT":
        client = OpenAI()
      else:
        self.model_id = "vikhyatk/moondream2"
        self.revision = "2024-08-26"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, revision=self.revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
      from openai import OpenAI

  def set_directory(self, path):
    if not os.path.exists(path):
      raise FileNotFoundError(f"The directory {path} does not exist.")
    else:
      self.directory = path

  def add_files_to_rag(self, file_name, file_path):
    if os.path.isfile(file_path):
      curr_images = convert_from_path(file_path)
      self.num_pages[self.doc_num] = len(curr_images)
      self.images = self.images + curr_images
      self.doc_num += 1
      self.RAG.index(
          input_path=file_path,
          index_name=file_name,  # index will be saved at index_root/index_name/
          store_collection_with_index=False,
          overwrite=True
      )
  
  def add_directory_to_rag(self, index_name):
    # index_name will be saved at index_root/index_name/
    self.RAG.index(
      input_path=self.directory,
      index_name="pdfs",
      store_collection_with_index=False,
      overwrite=True,
    )
    
  # Function to encode the image
  def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

  def query_text(self, text_query, k=1):
    return self.RAG.search(text_query, k=k)
  
  def query_image(self, image_query, text_query, k=1):
    enc_image = self.model.encode_image(image_query)
    return self.model.answer_question(enc_image, text_query, self.tokenizer)
  
  def prompt_gpt(self, client, image_bytes, text_query):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
      ]
    )
    print(completion.choices[0].message)

# Example usage

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

doc = paliDoc(list_ind=["pdfs"], model=None)
doc.set_directory("./pdfs")

# doc.add_directory_to_rag("pdfs")

text_query = "What are the course assignments and percentage of total grade of each?"
results = doc.query_text(text_query, k=1)

doc_num = results[0]['doc_id']
page_num = results[0]['page_num']

print(doc_num, page_num)
print(results[0])

pdfs = os.listdir(doc.directory)
target_image = convert_from_path(os.path.join(doc.directory, pdfs[doc_num]))[page_num - 1]

# print(doc.prompt_claude(target_image, text_query))
print(doc.query_image(target_image, text_query))


######
# Node.js
# Tailwind css


# directory = "./pdfs"

# if not os.path.exists(directory):
#     raise FileNotFoundError(f"The directory {directory} does not exist.")

# images = []
# num_pages = {}
# doc_num = 0;

# RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")

# for filename in os.listdir(directory):
#   f = os.path.join(directory, filename)

#   if os.path.isfile(f):
#     curr_images = convert_from_path(f)
#     num_pages[doc_num] = len(curr_images)
#     images = images + curr_images
#     doc_num += 1
#     RAG.index(
#         input_path=f,
#         index_name=filename, # index will be saved at index_root/index_name/
#         store_collection_with_index=False,
#         overwrite=True
#     )

# text_query = "How can I start writing reflection 1 for that ecology class?"
# results = RAG.search(text_query, k=1)

# model_id = "vikhyatk/moondream2"
# revision = "2024-08-26"
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, trust_remote_code=True, revision=revision
# )
# tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# doc_num = results[0]['doc_id']
# page_num = results[0]['page_num']

# for i in range(doc_num):
#   page_num += num_pages[i]

# target_image = images[page_num + 1]

# enc_image = model.encode_image(target_image)
# print(model.answer_question(enc_image, text_query, tokenizer))

# def set_directory(path):
#     return path

# def add_files_to_rag(directory, rag_model):
#     images = []
#     num_pages = {}
#     doc_num = 0

#     for filename in os.listdir(directory):
#         f = os.path.join(directory, filename)

#         if os.path.isfile(f):
#             curr_images = convert_from_path(f)
#             num_pages[doc_num] = len(curr_images)
#             images = images + curr_images
#             doc_num += 1
#             rag_model.index(
#                 input_path=f,
#                 index_name=filename,  # index will be saved at index_root/index_name/
#                 store_collection_with_index=False,
#                 overwrite=True
#             )
#     return images, num_pages

# def query_text(rag_model, text_query, k=1):
#     return rag_model.search(text_query, k=k)