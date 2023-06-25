import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Se non hai impostato la chiavi API nell'environment, togli il commento alle 2 righe seguenti e inserisci la tua chiave API
# import os
# os.environ["OPENAI_API_KEY"] = "..."

# Define system message for GPT
system_message = """Ti passerò la trascrizione di un video Youtube. Il tuo compito è creare degli script per una serie di "Youtube Shorts" correlati. Sta a te decidere quanti short creare, in base alle informazioni presenti nella trascrizione.

Uno Youtube Short è un video di 45-60 secondi che viene pubblicato su Youtube. È fatto per essere consumato da mobile, e per questo il formato è verticale ed è sempre accompagnato da captions per cho lo guarda senza audio.

Quando crei uno short, assicurati di seguire queste linee guida:

- La lunghezza del tuo script per lo short deve essere compresa tra 80 e 100 parole.
- Lo short deve contenere una singola idea ed esporla nella sua interità in maniera completa.
- Usa il framework "WIIFM", o "What's In It For Me": ogni short deve dare qualcosa di utile o interessante a chi lo guarda.
- Gli short devono matenere alta l'attenzione di un pubblico giovane e altamente distratto. Per questo, è importante che siano densi di contenuto e che contengano informazioni utili e interessanti dal primo all'ultimo secondo.
- Uno short non può essere troppo corto: deve esserci abbastanza tempo per comunicare un'idea in maniera chiara e completa e dare valore al pubblico. Per questo, cerca di scrivere almeno 80 parole.
- Ogni short deve comunicare una singola idea ed esporla nella sua interità: non creare shorts che richiedono di guardare altri shorts o il video principale per essere compresi.
- Per ogni short crea un titolo estremamente breve (massimo 6 parole), che verrà inserito come thumbnail allo short.
- Se la trascrizione non ha abbastanza materiale per generare uno short di qualità, il tuo output è una stringa vuota.
- Cerca di scrivere pochi Shorts, ma di alta qualità e densi di contenuto: la qualità è molto più importante della quantità."""

# Import text file
with open("trascrizione.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Create output .txt file
with open("output.txt", "w", encoding="utf-8") as file:
    file.write("")

def show_tokens(text):
    tokens = count_tokens(text)
    print(f"TOKENS: {tokens}")

def choose_model():
    """Scegli il modello da utilizzare. GPT-4 è consigliato, ma stranamente, anche 3.5 è in grado di dare risposte interessanti e diverse rispetto a 4."""
    print("Scegli il modello:")
    print("1. gpt-3.5-turbo-16k")
    print("2. gpt-4")
    model = input("Inserisci il numero del modello: ")
    if model == "1" or model == "":
        model = "gpt-3.5-turbo-16k-0613"
        limit = 10000
    elif model == "2":
        model = "gpt-4-0613"
        limit = 5000
    else:
        print("Modello non valido")
        exit()
    return model, limit

def count_tokens(text) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def split_text_in_chunks(tokens: int, limit):
    chunk_size = limit
    chunk_overlap = 100
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if tokens > limit:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            print("Diviso in blocchi")
            print(f"BLOCCO {i}: {count_tokens(chunk)} tokens")
    else:
        chunks = [text]
    return chunks

def write_output_for_chunk(chunk, model, sysmessage):
    chat = ChatOpenAI(temperature=0, model=model, client=any, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    messages = [
        SystemMessage(
            content=sysmessage
        ),
        HumanMessage(
            content=chunk
        ),
    ]
    completion = chat(messages)
    completion = completion.content
    # Write output to file
    with open("output.txt", "a", encoding="utf-8") as file:
        file.write(completion)
        file.write("\n\n")

def main():
    show_tokens(text)
    model, limit = choose_model()
    tokens = count_tokens(text)
    chunks = split_text_in_chunks(tokens, limit)
    for chunk in chunks:
        write_output_for_chunk(chunk, model, system_message)

if __name__ == "__main__":
    main()