# !pip install sentencepiece
modelname = "THUDM/chatglm-6b"
#model = "THUDM/chatglm2-6b"
#model = "/home/ubuntu/models/chatglm2-6b-int4"

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True) 
model = AutoModel.from_pretrained(modelname, trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
