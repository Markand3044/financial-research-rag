from langchain_huggingface import HuggingFaceEmbeddings

embadding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text = "Infosys reported strong revenue growth in fiscal 2026."

vector = embadding.embed_query(text)

print("Vector type :-",type(vector))
print("Vector length :-", len(vector))

print("\nFirst 10 values :- ")
print(vector[:10])