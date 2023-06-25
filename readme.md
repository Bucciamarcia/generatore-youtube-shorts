# Generatore Youtube Shorts

Generatore Youtube Shorts è un progetto open source che permette di creare degli script per una serie di "Youtube Shorts" correlati, basandosi sulla trascrizione di un video Youtube (che dovete creare separatamente - consiglio AWS Transcribe). Questo script genera shorts ottimizzati per avere alta concentrazione di contenuti utili e interessanti, attraverso l'uso dell'intelligenza artificiale.

## Avviso

Questo script è ancora molto basico, senza funzioni addizionali (come la possibilità di inserire argomenti manualmente per rendere l'AI più precisa).

I risultati non sono male, ma ancora imperfetti. Sentitevi liberi di modificare il prompt e, se ne trovate uno che funziona meglio di quello che ho messo io, suggerire modifiche.

### Requisiti

- Python (testato su 3.10, dovrebbe funzionare con tutte le versioni recenti)

### Installazione

1. Clonare il repository con `git clone` (o scaricarlo).

2. Entrare nella directory del progetto:

   ```
   cd generatore-youtube-shorts
   ```

3. Installare le dipendenze:

   ```
   pip install -r requirements.txt
   ```

### Utilizzo

1. Aggiungi la trascrizione del video Youtube nel file `trascrizione.txt`.
2. Langchain richiede di avere la chiave API nell'environment. Se non l'hai messa, allora de-commenta queste due righe in cima allo script e inserisci la tua chiave API di OpenAI:
```Python
import os
os.environ["OPENAI_API_KEY"] = "..."
```
3. Esegui lo script `main.py`:

   ```
   python main.py
   ```

4. Scegli il modello da utilizzare (GPT-4 è consigliato).
5. Il file di output con gli script per gli shorts verrà generato nel file `output.txt`.
6. Nota che ogni volta che si fa partire lo script, l'output precedente viene sovrascritto.

### Funzionalità principali

- **Scegliere il modello**: Lo script offre la possibilità di scegliere tra due modelli di intelligenza artificiale: GPT-3.5 e GPT-4.
- **Contare i token**: Lo script conta il numero di token presenti prima di procedere, così puoi interrompere lo script.
- **Suddividere il testo in parti**: Se il testo è troppo lungo, lo script lo divide in parti più piccole e le processa una alla volta.
- **Creare script per gli shorts**: Lo script genera script di 80-100 parole per gli shorts, tenendo conto delle linee guida specificate nel messaggio di sistema che può essere personalizzato.

### Contribuire

Se desideri contribuire al progetto, sei libero di aprire una issue o forkare il repository e creare una pull request. Tutti i contributi sono benvenuti!