Cool — I took a look at your repo. Here’s an improved README draft based on your existing structure and content, customized to your setup. You can copy this into your `README.md` and tweak as needed.

```markdown
# OAI

OAI is a collection of tiny, easy-to-train AI models. This project is **not** about matching giants like GLM-4.5 or GPT — it’s about understanding, experimenting, and learning.  

---

## 📂 Project Structure

```

OAI/
├── TLM/                 # Tiny Language Model implementation
├── TDM/                 # Tiny Diffusion Model implementation
├── data.txt             # (Optional) sample training data
├── README.md
└── LICENSE

````

---

## 🧠 Models

### TLM — Tiny Language Model  
A minimal LSTM-based language model in PyTorch.  
- Train on your own text  
- Generate text via temperature + top-k sampling  
- Save/restore model + vocabulary  
- Easy to read and hack  

### TDM — Tiny Diffusion Model  
A simple diffusion model for images (or synthetic data).  
- Basic forward and reverse diffusion  
- Adjustable noise schedules  
- Great for educational use  

*(If TDM isn't in the repo yet, add that section when it’s ready.)*

---

## 🚀 Usage

### Clone & Setup

```bash
git clone https://github.com/KeoLotso/OAI.git
cd OAI
pip install -r requirements.txt
````

### TLM

Run:

```bash
python TLM.py
```

You’ll be prompted whether to **train** a model or **interact** with an existing model.
During training, a sample is printed each epoch.

### TDM

```bash
python TDM.py
```

*(Adjust this command based on your actual TDM script filename and usage.)*

---

## 🧪 Example Output

Here’s what TLM might output after a few epochs:

```
SAMPLE: The night was dark and the stars ...
```

*(You can replace the above with a real sample from your model.)*

---

## 💡 Notes & Philosophy

* This is an educational / experimental project
* Prioritizes **clarity** and **simplicity** over performance
* You’ll learn more by modifying parts like the sampling method, architecture, or hyperparameters

---

## 🤝 Contributing

Have ideas for a new tiny model? Want to improve sampling, add attention, or try another architecture? Open an issue or send a PR — I’d love to see what you do.

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and build on it.
