# 🤖 GenAI and Simple LLM Inference on CPU with Fine-Tuning for a Custom Chatbot

## 📌 Project Overview  
This project focuses on **fine-tuning a large language model (LLM) to create a custom chatbot** using readily available hardware, specifically **4th Generation Intel® Xeon® Scalable processors**. The model is fine-tuned on the **Alpaca Dataset** using **Llama 2** from Meta, leveraging **Intel Developer Cloud (IDC)** for efficient training and inference.

---

## 🎯 Objectives  
✅ **Fine-tune an LLM** to create a custom chatbot.  
✅ **Utilize Intel Developer Cloud (IDC)** for model training and inference.  
✅ **Optimize fine-tuning** using the **Alpaca Dataset** and **Llama 2 models**.  

---

## 📂 Dataset  
The **Alpaca Dataset** from Stanford University serves as the primary dataset for fine-tuning. It consists of:  
- **175 seed tasks** expanded into **52K diverse instruction-response pairs**.  
- **JSON format** dataset designed for instruction-following tasks.  

---

## 🧠 Model  
The project utilizes **Llama 2**, a family of pre-trained and fine-tuned **large language models (LLMs) by Meta**, ranging from **7B to 70B parameters**. The model is fine-tuned to enhance conversational capabilities.

---

## 🛠️ Development Platform  
The project is executed on **Intel Developer Cloud (IDC)**, which provides:  
- **High-performance GPUs** and **enterprise-grade CPUs**.  
- **Latest Intel hardware and software optimizations** for AI/ML workloads.

---

## 🔧 Tools & Technologies  
🔹 **Intel® Xeon® Scalable Processors**  
🔹 **Intel® Extension for Transformers’ Neural Chat**  
🔹 **Alpaca Dataset**  
🔹 **Llama 2 Models**  
🔹 **Intel Developer Cloud (IDC)**  

---

## 🚀 Installation & Setup  
### **1️⃣ Install Required Libraries**  
```bash  
pip install transformers datasets torch requests  
```

### **2️⃣ Clone the Repository**  
```bash  
git clone https://github.com/intel/intel-extension-for-transformers  
```

### **3️⃣ Navigate to the Relevant Directory**  
```bash  
cd intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/docs/notebooks  
```

---

## ⚙️ Implementation Steps  
### **Step 1: Load the Alpaca Dataset**  
```python  
import json, requests
url = 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json'  
response = requests.get(url)  
alpaca_data = response.json()  

# Save the dataset locally  
with open('alpaca_data.json', 'w') as f:  
    json.dump(alpaca_data, f)  
```

### **Step 2: Load the Llama 2 Model**  
```python  
from transformers import AutoModelForCausalLM, AutoTokenizer  
model_name = "meta-llama/Llama-2-7b-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForCausalLM.from_pretrained(model_name)  
```

### **Step 3: Fine-Tune the Model**  
```python  
from transformers import Trainer, TrainingArguments  
import torch  

# Preprocess Dataset  
def preprocess(data):  
    inputs = tokenizer(data['instruction'], return_tensors='pt')  
    labels = tokenizer(data['response'], return_tensors='pt')['input_ids']  
    return inputs['input_ids'], labels  

train_data = [preprocess(sample) for sample in alpaca_data]  

# Define Training Arguments  
training_args = TrainingArguments(  
    output_dir='./results',  
    num_train_epochs=3,  
    per_device_train_batch_size=2,  
    warmup_steps=500,  
    weight_decay=0.01,  
    logging_dir='./logs',  
    logging_steps=10,  
)

# Create Custom Dataset Class  
class CustomDataset(torch.utils.data.Dataset):  
    def __init__(self, data):  
        self.data = data  
    def __len__(self):  
        return len(self.data)  
    def __getitem__(self, idx):  
        return {'input_ids': self.data[idx][0], 'labels': self.data[idx][1]}  

train_dataset = CustomDataset(train_data)  

# Initialize Trainer  
trainer = Trainer(  
    model=model,  
    args=training_args,  
    train_dataset=train_dataset,  
)

# Train the Model  
trainer.train()  
```

### **Step 4: Save the Fine-Tuned Model**  
```python  
model.save_pretrained('./fine_tuned_model')  
tokenizer.save_pretrained('./fine_tuned_model')  
```

### **Step 5: Define Inference Function**  
```python  
def generate_response(prompt):  
    inputs = tokenizer(prompt, return_tensors='pt')  
    outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)  
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  
```

### **Step 6: Example Inference**  
```python  
print(generate_response("Hello, how can I help you today?"))  
print(generate_response("What is the capital of France?"))  
print(generate_response("Tell me a joke."))  
```

---

## 📊 Evaluation  
### **Performance**  
🔹 **Hardware Used:** 4th Generation Intel® Xeon® Scalable processors  
🔹 **Training Time:** Varies based on model size and dataset.  

### **Classification Performance**  
✔️ **Works Well For:** Simple and factual queries (greetings, weather, facts).  
❌ **Struggles With:** Complex queries requiring deep reasoning or external knowledge.  

### **Analysis & Problem-Solving**  
One challenge was **irrelevant or repetitive responses**. This was addressed by:  
- Further fine-tuning with additional data.  
- Adjusting **hyperparameters** for improved response quality.  

---

## 📌 Conclusion  
This project **demonstrates the feasibility of fine-tuning a large language model** to create a custom chatbot using Intel's **advanced hardware and software tools**. The **Alpaca Dataset** and **Intel® Extension for Transformers’ Neural Chat** helped build a functional chatbot capable of handling diverse queries. Future improvements can focus on:  
✅ Expanding the dataset.  
✅ Further optimizing fine-tuning for better conversational accuracy.  

---

## 📜 References  
- **Intel Extension for Transformers - Neural Chat**  
- **Alpaca Dataset from Stanford University**  
- **Intel Developer Cloud & AI Tools**  

---

## 📩 Acknowledgments  
Thank you to all **team members and mentors** for their guidance and support in this project.  
