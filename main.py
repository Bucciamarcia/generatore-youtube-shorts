import tiktoken
from openai import OpenAI
from langchain.text_splitter import TokenTextSplitter

# Se non hai impostato la chiavi API nell'environment, togli il commento alle 2 righe seguenti e inserisci la tua chiave API
# import os
# os.environ["OPENAI_API_KEY"] = "..."

MODEL = "gpt-4o"


def extract_text(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    return text


system_message = extract_text("system_message.txt")

# Import text file
with open("trascrizione.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Create output .txt file
with open("output.txt", "w", encoding="utf-8") as file:
    file.write("")


def show_tokens(text):
    tokens = count_tokens(text)
    print(f"TOKENS: {tokens}")


def count_tokens(text) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def split_text_in_chunks(tokens: int, limit):
    chunk_size = limit
    chunk_overlap = 100
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    print("Divido in blocchi...")
    if tokens > limit:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            print(f"BLOCCO {i}: {count_tokens(chunk)} tokens")
    else:
        chunks = [text]
    return chunks


def write_output_for_chunk(chunk, model, sysmessage):
    client = OpenAI()
    output = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sysmessage},
            {"role": "user", "content": chunk},
        ],
    )
    completion = output.choices[0].message.content
    # Write output to file
    with open("output.txt", "a", encoding="utf-8") as file:
        file.write(completion)
        file.write("\n\n")


def main():
    show_tokens(text)
    limit = 100000
    tokens = count_tokens(text)
    chunks = split_text_in_chunks(tokens, limit)
    for chunk in chunks:
        write_output_for_chunk(chunk, MODEL, system_message)


if __name__ == "__main__":
    main()
