import json

def extract_code():
    nb_path = r"c:\Users\shiva\OneDrive\Desktop\GAN_Project\code\GAN_FashionMNIST_11_03.ipynb"
    out_path = r"c:\Users\shiva\OneDrive\Desktop\GAN_Project\code\extracted_model.py"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    with open(out_path, "w", encoding="utf-8") as out:
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                out.write("".join(cell["source"]) + "\n\n")

if __name__ == "__main__":
    extract_code()
