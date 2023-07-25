import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Se non hai impostato la chiavi API nell'environment, togli il commento alle 2 righe seguenti e inserisci la tua chiave API
# import os
# os.environ["OPENAI_API_KEY"] = "..."

# Define system message for GPT
system_message = """Ti passerò la trascrizione di un video Youtube. Il tuo compito è creare una serie di script per degli "Youtube Shorts" che estraggono i punti più interessanti del video pricinpale sotto forma di brevi contenuti. Sta a te decidere quanti short creare, in base alle informazioni presenti nella trascrizione.

Uno Youtube Short è un breve video indipendente di 45-60 secondi che viene pubblicato su Youtube. È fatto per essere consumato da mobile, e per questo il formato è verticale ed è sempre accompagnato da captions per cho lo guarda senza audio.

Quando crei uno short, assicurati di seguire queste linee guida:

- Ciascuno short deve essere compresa tra 80 e 100 parole.
- Lo short deve contenere una singola idea ed esporla in maniera completa ed incisiva.
- Il payoff deve sempre essere incluso nello Short.
- Usa il framework "WIIFM", o "What's In It For Me": ogni short deve dare qualcosa di utile o interessante a chi lo guarda.
- Gli short devono matenere alta l'attenzione di un pubblico giovane e altamente distratto. Per questo, è importante che siano densi di contenuto e che contengano informazioni utili e interessanti dal primo all'ultimo secondo, senza perdersi in introduzioni o saluti.
- Uno short deve essere corto ma deve comunicare un'idea in maniera chiara e completa e dare valore al pubblico. Per questo, cerca di scrivere almeno 80 parole.
- Ogni Short deve essere a sé stante e contenere un'idea nella sua interità: non fare mai riferimento ad altri short, video o contenuti.
- Vedi uno Short come se fosse un TikTok.
- Fai solo contenuti di qualità: non creare short solo per creare short. Se non c'è abbastanza materiale interessante nella trascrizione, non creare short o creane di meno.
- Per ogni short crea un titolo estremamente breve (massimo 6 parole) e intrigante.
- Crea contenuti intriganti, stimolanti e accattivanti.
- Le due metriche più importanti sono ritenzione del pubblico e "mi piace" dati allo Short.
- Scrivi in lingua italiana.

ESEMPI

CONTENUTO SBAGLIATO:

"Ti sei mai chiesto se un'intelligenza artificiale può avere principi morali? Ho condotto un esperimento con GPT-4 per scoprirlo. Ho posto a GPT-4 dei dilemmi morali per vedere come risponde. I risultati sono stati sorprendenti. Nonostante sia un computer, GPT-4 ha simulato una serie di principi morali. Scopriamo insieme quali sono!"

MOTIVO: In questo Short manca il payoff: lo short si interrompe qui, senza dare al pubblico un contenuto di qualità. Visto che terminato lo short non c'è altro da guardare ,non c'è niente da "scoprire insieme".

CONTENUTO CORRETTO:

"Ho condotto un esperimento con GPT-4 per scoprire se un'intelligenza artificiale può simulare principi morali. Nonostante sia un computer, GPT-4 ha simulato una serie di principi morali ineccepibili. Per questo sono convinto che in questo momento GPT-4 sia in grado di prendere decisioni morali migliori di quelle di molti esseri umani, molti dei quali ci governano."

MOTIVO: In questo Short il payoff è chiaro: GPT-4 è in grado di prendere decisioni morali migliori di quelle di molti esseri umani, molti dei quali ci governano. Questo è un payoff interessante e intrigante, che stimola la curiosità del pubblico. L'obiettivo non è di portare il pubblico al video principale, ma a dargli una soddisfazione immediata abbastanza da mettere un "mi piace" al video. Inoltre, questo short è della lunghezza corretta e va dritta al punto.

CONTENUTO SBAGLIATO:

"Secondo GPT-4, la prima cosa da fare per migliorare un regime dispotico sarebbe allentare i controlli sui media e su internet, consentendo di ascoltare una gamma più diversificata di voci. Ciò comporterebbe anche la fine della persecuzione di coloro che criticano il governo, il che manderebbe un segnale che il dissenso non è più vista come una minaccia."

MOTIVO: Questo short è troppo lungo. Inoltre, il payoff non è molto forte, e l'idea che comunica è troppo generica e non è molto intrigante. Manca un contesto per far capire al pubblico di cosa si sta parlando.

CONTENUTO CORRETTO:

"Ho chiesto a GPT-4 cosa farebbe se fosse a capo di un regime dispotico. Secondo lui, la prima cosa da fare sarebbe allentare i controlli sui media e su internet, consentendo di ascoltare una gamma più diversificata di voci. Questo è meglio di quello che molti leader di molti Paesi stanno facendo in questo momento, e per questo sono convinto che GPT-4 abbia una moralità superiore a quella di molte persone che in questo momento ci governano."

MOTIVO: Questo short è della lunghezza corretta e va dritto al punto. Inoltre, il payoff è chiaro e intrigante: GPT-4 è in grado di prendere decisioni morali migliori di quelle di molti esseri umani. Visto che gli Shorts non sono correlati fra di loro, devi sempre spiegare di cosa stai parlando, anche se hai già fatto altri Shorts sullo stesso argomento."""

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
        limit = 8000
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
    print("Divido in blocchi...")
    if tokens > limit:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            print(f"BLOCCO {i}: {count_tokens(chunk)} tokens")
    else:
        chunks = [text]
    return chunks

def write_output_for_chunk(chunk, model, sysmessage):
    chat = ChatOpenAI(temperature=0.3, model=model, client=any, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
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