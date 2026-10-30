# Valutazione del RAG UAP: metodo, criteri e lettura delle metriche

Questo documento racconta come misuro il sistema, come decido cosa modificare e come leggere i numeri che produce. Non è un manuale d'uso (quello è nel `README.md`): è il registro del ragionamento dietro le scelte, pensato per poter riprendere il lavoro a distanza di tempo senza dover ricostruire tutto.

---

## 1. Da dove sono partito e perché

### La v1 in breve

Il sistema segue questo percorso:

1. PDF convertiti in Markdown.
2. Chunking che rispetta la struttura (titoli, tabelle), con chunk da massimo 1200 caratteri e 150 di overlap.
3. Embedding con `text-embedding-3-small` (1536 dimensioni).
4. Indice su Azure AI Search, con campo testuale e campo vettoriale (HNSW).
5. Ricerca ibrida (keyword + vettoriale, fusione RRF).
6. Risposta generata da `gpt-5-mini`, vincolata al contesto e con citazione delle fonti `[1]`, `[2]`, ...

Il corpus sono 8 documenti in italiano sugli UAP.

### Perché misurare prima di cambiare

Un RAG può sbagliare in due punti diversi: può **non recuperare** i passaggi giusti, oppure può **usarli male**. Se non separo i due casi rischio di ritoccare il prompt per un problema che in realtà sta nell'indice, o viceversa.

Per questo ho costruito due valutazioni distinte, entrambe sullo stesso dataset:

- **Retrieval**: la sezione giusta compare tra i risultati, e in che posizione?
- **Generazione**: la risposta prodotta è fedele al contesto, pertinente e corretta rispetto a una risposta di riferimento?

Regola di lavoro: se il retrieval è scarso, è inutile intervenire sul prompt.

### Il dataset

`evaluation/dataset.json` contiene 21 domande. Per ciascuna ho indicato la risposta di riferimento e le sezioni dei documenti in cui si trova.

| Tipo | Numero | Scopo |
|---|---|---|
| `direct` | 13 | Domande con le parole del documento |
| `paraphrase` | 3 | Stesso contenuto, parole diverse |
| `multi_hop` | 2 | La risposta richiede più sezioni |
| senza risposta | 3 | Verificare che il sistema ammetta di non sapere |

Le 3 domande senza risposta non rientrano nelle metriche numeriche: le controllo a mano leggendo cosa risponde il sistema (sezione 4.3).

### Baseline di retrieval (`top_k=10`, 18 domande con risposta)

| Metodo | hit@1 | hit@3 | hit@10 | recall@10 | MRR@10 | copertura piena |
|---|---|---|---|---|---|---|
| keyword | 0.39 | 0.72 | 0.72 | 0.72 | 0.56 | 0.72 |
| vector | 0.44 | 0.61 | 0.78 | 0.75 | 0.54 | 0.72 |
| **hybrid** | **0.50** | **0.67** | **0.83** | **0.81** | **0.60** | **0.78** |

Come la leggo:

- La ricerca ibrida batte entrambe le altre ed è quella che il sistema usa.
- Il contenuto giusto, in molti casi, c'è (hit@10 = 0.83), ma **solo una volta su due è al primo posto** (hit@1 = 0.50). Qui c'è margine per un riordino dei risultati.
- Per tipo di domanda, il recall in hybrid è 0.92 sulle `direct`, 0.75 sulle `multi_hop` e **0.33 sulle `paraphrase`**. Il punto debole sono le domande formulate con parole diverse da quelle del documento.

### Baseline di generazione (18 domande con risposta)

Il giudice è un modello Azure; le metriche sono descritte nella sezione 4.

| Metrica | Media | Minimo |
|---|---|---|
| faithfulness | **0.95** | 0.80 |
| answer_relevancy | **0.74** | 0.43 |
| answer_correctness | **0.62** | 0.14 |

Per tipo di domanda:

| Tipo | n | faithfulness | answer_relevancy | answer_correctness |
|---|---|---|---|---|
| direct | 13 | 0.94 | 0.73 | 0.67 |
| paraphrase | 3 | 1.00 | 0.73 | **0.37** |
| multi_hop | 2 | 0.89 | 0.85 | 0.65 |

Le parafrasi mostrano lo stesso schema del retrieval: la fedeltà è perfetta, la correttezza è bassa. Il sistema è fedele a un contesto che però non contiene la sezione giusta.

---

## 2. Cosa ho osservato

Prima di modificare qualcosa ho guardato **cosa occupava i posti** nei risultati, non solo quante volte la sezione giusta compariva.

### Dal retrieval

- **Contenuto fuori tema nell'indice.** Due blocchi del documento sulle release recenti non parlano di UAP: una guida su come scrivere documenti per un RAG e una sezione di idee per un marchio di abbigliamento. Compaiono tra i primi 10 risultati di **15 domande su 18** e occupano **23 posti su 180**. In tre delle quattro domande sbagliate sono tra i primi tre risultati. Hanno anche contaminato una risposta (un termine presente solo in quei blocchi è finito nell'elenco dei documenti ufficiali, facendo scendere la faithfulness a 0.76 in una prova).
- **Testo non informativo in coda ai documenti.** Tutti gli 8 documenti terminano con una riga di generazione automatica, e 4 hanno anche un invito a scaricare il contenuto. Finiscono nell'ultimo chunk di ogni documento, con una data (2026) che attira le domande temporali.
- **Chunk di solo titolo.** I chunk 0 con titolo e sottotitolo del documento sono generici e risalgono nelle classifiche vettoriali (in un caso erano **primi** per una domanda sul 2004).
- **Documenti piccoli quasi invisibili.** I PDF più brevi (2, 3 e 4) occupano 22 posti su 180 nei risultati; la Nota metodologica, le Release e il Glossario ne occupano 123. Nei PDF piccoli il titolo del documento è una sezione vuota che viene scartata, quindi nessun chunk contiene più il nome del documento (per esempio "Casistiche").
- **Contenuti duplicati.** Gli stessi fatti (Blue Book, GEIPAN, ciclo di gestione, criteri di un caso "forte") compaiono in più documenti. Il dataset accetta di solito una sola sezione, quindi il retrieval può essere penalizzato per una risposta corretta (succede per Q01 e Q11).
- **Ricerca per parole chiave senza analisi linguistica.** Il campo testuale usa l'analizzatore di default, senza stemming né stopword italiane. Su domande piene di parole comuni vince il documento più ricco di parole generiche.
- **La domanda sugli eventi 2004-2022 (Q21).** I due chunk che contengono la risposta esistono e sono ben formati, ma in hybrid si piazzano all'8° e all'11° posto. Il problema è la classifica, non il chunking: per questo non cambio la dimensione dei chunk.

### Dalla generazione

Sei risposte hanno una correttezza sotto 0.50, ma non hanno la stessa causa.

- **Guasti veri, causati dal retrieval.** Q02 (0.14) e Q04 (0.26): il contesto non contiene la sezione "classificazione per caratteristica osservata", e il modello descrive un'altra classificazione. Q21 (0.54): la risposta elenca 2020, 2021 e 2022 e salta Nimitz 2004, AATIP e il 2017, perché il chunk con quegli eventi non è arrivato.
- **Risposte corrette penalizzate dalla metrica.** Q14 (0.36), Q11 (0.45), Q07 (0.47), Q15 (0.48): in Q14 la risposta coincide con il riferimento e aggiunge solo brevi descrizioni, tutte vere. La metrica conta come "non nel riferimento" ogni dettaglio in più, quindi i riferimenti molto sintetici penalizzano le risposte più complete.
- **Un riferimento sbagliato.** In Q08 (0.57) il riferimento dice che il documento "non definisce B e C" della scala GEIPAN, ma la Nota metodologica e il Glossario le definiscono. La risposta del sistema è corretta, il riferimento no.

Conseguenza: la correttezza media (0.62) **sottostima** il sistema. La uso come misura comparativa tra versioni, non come valore assoluto.

### Le domande senza risposta

Su 3 domande senza risposta nel corpus, il sistema ha ammesso correttamente di non sapere in **3 casi su 3**: non ha inventato il nome di un alieno a Roswell, non ha dato una percentuale GEIPAN 2025 inesistente e non ha fornito coordinate GPS non presenti nei documenti. In un caso chiude invitando l'utente a fornire un documento, una frase che posso eliminare con una riga nel prompt.

Limite del test: il dataset non contiene una vera domanda con **premessa falsa** né una completamente fuori dominio. Le aggiungerò.

---

## 3. Cosa cambio nella v2 e con quale criterio

### Il criterio

Una modifica entra nella v2 solo se soddisfa tre condizioni:

1. **Ha una causa osservata nei dati** (una delle evidenze della sezione 2), non una convinzione generica.
2. **È isolabile**: posso attivarla da sola e misurarne l'effetto.
3. **Ha un costo basso** rispetto al beneficio atteso.

### Interventi

| Intervento | Evidenza che lo motiva | Perché lo faccio così |
|---|---|---|
| Esclusione per titolo di tre sezioni non pertinenti (guida RAG e idee per il marchio) | 15/18 query contaminate, 23/180 posti | Il Markdown viene rigenerato dal PDF a ogni esecuzione; un elenco esplicito di sezioni escluse è tracciabile e reversibile |
| Filtro della riga di generazione automatica nel chunker | Presente in tutti gli 8 documenti | Riguarda tutti i file, quindi lo gestisco una volta sola nel codice |
| Esclusione dei chunk di solo titolo | Chunk 0 generici nei primi posti | Aggiungono somiglianza con qualsiasi domanda, ma nessuna informazione |
| Analizzatore italiano sui campi testuali | Ricerca keyword guidata da parole comuni | Stemming e stopword rendono la ricerca lessicale utile sull'italiano |
| Prefisso di contesto negli embedding ("documento > sezione") | Documenti piccoli sotto-rappresentati, parafrasi a 0.33 | Il vettore "sa" di quale documento e sezione è il chunk; il testo mostrato al modello non cambia |
| Metadati puliti: categorie per tutti i file, titoli senza formattazione, `source_file` come nome file | 4 documenti su 8 senza categoria | Il filtro per categoria funziona solo se tutti i documenti ne hanno una |
| Cancellazione dei chunk di un documento prima di reindicizzarlo (una volta per documento) | Un nuovo caricamento aggiorna ma non rimuove | Senza cancellazione i vecchi chunk restano nell'indice anche dopo averli tolti dai file |
| Allineamento di `top_k` a 10 in tutto il codice | Valori diversi in punti diversi (5, 10, 20) | La valutazione deve usare gli stessi parametri della produzione |
| Rimozione di un riferimento a un riordinatore (reranker) non ancora implementato | Faceva fallire la funzione `ask()` | Codice morto: lo reintrodurrò quando sarà davvero implementato |

**Compromesso noto.** La sezione sulle idee per il marchio contiene anche una "nota sulle fonti" che spiega quali fonti sono non ufficiali. Con l'esclusione per titolo la perdo in v2a. Se le domande sulla distinzione tra fonti ufficiali e non ufficiali peggiorano, la ripristino spostandola nella sezione "Fonti" del documento sorgente.

### In che ordine, per isolare gli effetti

Dividerò la v2 in due passaggi, entrambi con indice ricreato da zero (il costo di embedding è di pochi centesimi):

- **v2a**: pulizia dei dati, filtro, chunk-titolo, analizzatore italiano, metadati.
- **v2b**: v2a più il prefisso di contesto negli embedding.

Così, se v2b migliora rispetto a v2a, so che il merito è del prefisso.

### Cosa lascio fuori, di proposito

- **Modifiche al dataset** (riferimento di Q08, riferimenti più completi, sezioni alternative per i duplicati). Lo congelo fino alla v2b per confrontare versioni con lo stesso metro; poi ricalcolo la baseline offline con il dataset corretto.
- **Un chunk per ogni evento della cronologia.** Per domande come "eventi tra il 2004 e il 2022" i due chunk grandi di oggi contengono già tutta la risposta; spezzarli la renderebbe più difficile da ricomporre.
- **Il riordino dei risultati con il semantic ranker di Azure.** È il passo successivo, da misurare separatamente dopo la pulizia.

---

## 4. Come leggere le valutazioni

### 4.1 Metriche di retrieval

Calcolate sulle sezioni attese (coppia file + titolo). Per ogni domanda il valore va da 0 a 1; nei riepiloghi è la media sulle domande.

| Metrica | Range | Significato |
|---|---|---|
| `hit@1` | 0-1 | La prima sezione recuperata è una di quelle attese |
| `hit@3`, `hit@10` | 0-1 | Una sezione attesa compare nei primi 3 / 10 risultati |
| `recall@10` | 0-1 | Sezioni attese trovate / sezioni attese totali |
| `mrr@10` | 0-1 | 1 / posizione della prima sezione attesa (pos. 1 = 1.0, pos. 2 = 0.5, pos. 10 = 0.1) |
| `full_coverage@10` | 0 o 1 | Trovate **tutte** le sezioni attese (conta per le domande multi-sezione) |

Un 0.83 su `hit@10` con 18 domande significa 15 domande su 18.

Ordine di lettura: prima `hit@10` e `recall@10` (il contenuto è stato trovato?), poi `hit@1` e `MRR` (è in cima?).

### 4.2 Metriche di generazione (giudice: un modello di linguaggio)

Il giudice è un modello Azure che valuta ogni risposta. Tutte e tre le metriche sono pensate tra 0 e 1.

**Faithfulness (fedeltà al contesto).** La risposta viene scomposta in affermazioni; per ognuna si verifica se è deducibile dai passaggi recuperati. Punteggio = affermazioni supportate / affermazioni totali.

- **1.00**: ogni affermazione è supportata dal contesto.
- **0.89**: circa 8 affermazioni su 9 sono supportate, una no.
- **0.50**: metà della risposta non è ancorata ai documenti.

Dice se il modello ha *inventato o aggiunto*. Non dice se la risposta è giusta: una risposta può essere fedele a un contesto sbagliato.

**Answer relevancy (pertinenza).** Il giudice genera alcune domande a partire dalla risposta e ne misura la somiglianza semantica con la domanda originale. Se la risposta è centrata, le domande generate somigliano a quella reale.

- **0.89**: la risposta è centrata e completa rispetto alla domanda.
- **0.60**: risposta in parte fuori fuoco, incompleta o molto sintetica.
- **sotto 0.40**: o fuori tema, o (più spesso) troppo breve.

Dice se la risposta *parla di ciò che è stato chiesto*, non se è vera. È la metrica meno stabile: le risposte brevi e a elenco, anche perfette, prendono punteggi bassi e cambiano da un'esecuzione all'altra (sulla stessa domanda ho visto 0.37, 0.42 e 0.59). La uso come segnale secondario.

**Answer correctness (correttezza).** Confronta la risposta con quella di riferimento: 75% accordo sui fatti (affermazioni corrette, errate, mancanti) e 25% somiglianza semantica.

- **1.00**: stessi fatti del riferimento.
- **0.77**: fatti corretti, formulati o completati in modo diverso.
- **0.14**: la risposta racconta altro rispetto al riferimento.

È la metrica più vicina a "è giusta?", ma dipende dalla qualità della risposta di riferimento che ho scritto: un riferimento troppo sintetico penalizza le risposte più ricche (vedi Q14), uno sbagliato penalizza risposte corrette (vedi Q08). Per questo la leggo in modo comparativo.

Soglie di lavoro (le uso io per orientarmi, non sono uno standard): ≥ 0.90 ottimo, 0.75-0.90 buono, 0.50-0.75 da guardare, < 0.50 problema.

### Come incrociare le tre metriche

| Quadro | Probabile causa |
|---|---|
| faithfulness bassa | Il modello aggiunge informazioni che non sono nel contesto, magari prese da un chunk di rumore |
| faithfulness alta, correctness bassa | Il retrieval ha portato il contesto sbagliato (Q02, Q04), oppure il riferimento è troppo sintetico o errato (Q14, Q08) |
| tutte basse | Contesto sbagliato e risposta divagante: partire dal retrieval |
| relevancy bassa, altre alte | Quasi sempre una risposta corretta ma molto breve: non è un problema |
| tutte alte | Il sistema funziona su quella domanda |

Quando la correctness è bassa con faithfulness alta, il primo controllo è sempre lo stesso: leggere l'elenco dei chunk recuperati e verificare se la sezione attesa c'è.

### 4.3 Le domande senza risposta

Per le domande senza risposta nel corpus (informazione assente, argomento fuori dominio, premessa falsa) leggo direttamente la risposta e controllo che:

- ammetta che il contesto non basta;
- non inventi dettagli;
- se la premessa è sbagliata, la contesti invece di assecondarla.

Esito sulla baseline: 3 su 3 corrette. Un'eventuale metrica automatica di astensione la aggiungerò dopo la v2.

### 4.4 Limiti da tenere presenti

- **Dataset piccolo.** Con 18 domande, una sola vale circa 5.5 punti percentuali. Una differenza di una domanda tra due versioni è rumore.
- **Poche parafrasi.** Sono solo 3, quindi il dato sulle parafrasi è fragile.
- **Risultati non deterministici.** I modelli usati non permettono di fissare la temperatura: la stessa domanda può dare risposte e punteggi leggermente diversi. Per confrontare due versioni guardo le **variazioni nette sulle singole domande** (da fallita a riuscita e viceversa) più che la media.
- **Riferimenti imperfetti.** Alcuni sono troppo sintetici e uno (Q08) è incompleto: la correttezza assoluta va letta con cautela.
- **Il retrieval è misurato per sezione**, non per singolo chunk, e non verifica che nel chunk ci sia davvero la risposta.
- **Duplicati tra documenti** possono far sembrare un errore una risposta corretta (vedi Q01 e Q11).

---

## 5. Come procedo da qui

Ordine dei passi:

1. ~~Baseline di retrieval e di generazione sulle 18 domande~~ (completata, sezione 1).
2. **v2a**: pulizia dei dati, filtro, chunk-titolo, analizzatore italiano, metadati. Ricreare l'indice e rilanciare le due valutazioni con `--label v2a`.
3. **v2b**: aggiungere il prefisso di contesto negli embedding, ricreare l'indice e rilanciare con `--label v2b`.
4. **Confronto** tra baseline, v2a e v2b, guardando sia le medie sia le singole domande che cambiano stato (in particolare Q02, Q04 e Q21).
5. **Sistemare il dataset**: correggere il riferimento di Q08, rendere più completi i riferimenti troppo sintetici, indicare le sezioni alternative per i contenuti duplicati, aggiungere almeno 8-10 parafrasi e le domande con premessa falsa e fuori dominio. Ricalcolare le baseline offline con le nuove regole, per un confronto equo.
6. **Provare il semantic ranker** di Azure AI Search: recuperare 20-30 candidati, riordinarli e tenerne 6-8. Si valuta come un passaggio separato.
7. **Valutazione dell'astensione** sulle domande senza risposta.
8. **Esporre il servizio** con un'API (FastAPI) solo dopo aver stabilizzato qualità e metriche.

Criterio di accettazione di una modifica: resta se migliora `hit@10`/`recall@10` o `answer_correctness` **senza far peggiorare domande che prima funzionavano**. Controllo il costo nel portale Azure dopo le prime esecuzioni complete.

### Registro dei risultati

| Versione | hit@1 | hit@10 | recall@10 | MRR@10 | faithfulness | answer_relevancy | answer_correctness |
|---|---|---|---|---|---|---|---|
| baseline (v1) | 0.50 | 0.83 | 0.81 | 0.60 | 0.95 | 0.74 | 0.62 |
| v2a | — | — | — | — | — | — | — |
| v2b | — | — | — | — | — | — | — |

(Valori di retrieval in modalità hybrid, `top_k=10`. Generazione su 18 domande con risposta.)

---

## 6. Come riprodurre le misure

```powershell
# Ingestion (crea/aggiorna l'indice)
uv run python -m uap_project.main

# Valutazione del retrieval (salva retrieval_<label>_top_<k>.json in evaluation/results/)
uv run python -m uap_project.evaluation.evaluate_retrieval --top-k 10 --label baseline

# Valutazione della generazione (salva generation_eval_<timestamp>.json)
uv run python -u -m uap_project.evaluation.evaluate_generation

# Diagnosi di una singola domanda: dove si piazzano le sezioni attese
uv run python -m uap_project.evaluation.debug_query "domanda" --top-k 20 --mark "testo sezione"
```

Nei risultati JSON, per ogni domanda trovo i chunk recuperati con posizione e punteggio: è il punto di partenza per capire *perché* una domanda fallisce, prima ancora di guardare le medie.