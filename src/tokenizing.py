from transformers import AutoTokenizer

def config_tokenizer(model, df, col_name, padding="longest", truncation=True, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer(df[col_name], padding=padding, truncation=truncation, max_length=max_length)
