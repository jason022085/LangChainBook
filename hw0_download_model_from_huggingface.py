import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login
# Log in
from dotenv import load_dotenv, find_dotenv
# 加载当前目录或上级目录中的 .env 文件
load_dotenv(find_dotenv())
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
login(token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
# 加载 HuggingFace 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# 使用 GPU（device=0 表示使用第一張 GPU）
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer , device=0)
# 使用 HuggingFacePipeline 将 HuggingFace 模型集成到 LangChain
llm = HuggingFacePipeline(pipeline=pipe)
prompt = "What is the capital of China?"
result = llm.invoke(prompt)
print(result)