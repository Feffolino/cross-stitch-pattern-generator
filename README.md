# Cross-stitch pattern generator

**EN** — Turn pixel art, cross-stitch charts found online, or ordinary photos into a
clean, printable cross-stitch pattern: automatic grid detection, background removal,
black-and-white "ink" line-art style or reduced colors, and a PDF split into A4 sheets
with row/column numbers.

**IT** — Generatore di schemi a punto croce da pixel art, schemi trovati online o foto.

Trasforma un'immagine pixel art presa online in uno schema per punto croce:
trova da solo la griglia dei quadrati, ridisegna l'immagine pulita con le linee
sui bordi e crea un PDF da stampare diviso in fogli.

Funziona anche con immagini ingrandite di un fattore non intero, sfocate,
compresse JPEG, con margini o con la griglia gia' disegnata (screenshot dal
telefono: le fasce attorno vengono tolte da sole).

## Requisiti

Python 3 con `numpy`, `Pillow`, `scipy` e, per togliere lo sfondo alle foto, `rembg`:

```
pip install numpy pillow scipy "rembg[cpu]==2.0.69"
```

rembg 2.0.69 funziona con numpy 2.2 / Pillow 11; le versioni piu' nuove richiedono
numpy >= 2.3 e Pillow >= 12, che possono entrare in conflitto con altri pacchetti.
Il modello `u2net` (176 MB) viene scaricato al primo uso in `~/.u2net`.

## Uso veloce

```
python gridify.py immagine.jpg
```

Il programma capisce da solo se l'immagine e' uno schema gia' fatto (lo ridisegna
pulito), pixel art salvata a grandezza naturale (1 pixel = 1 quadrato, fino a 300
quadrati per lato) oppure una foto normale (la trasforma in schema). Le scritte
lungo un bordo (nome del sito) negli schemi trovati online vengono tolte.

### Foto normali

Tutto automatico:

1. **Sfondo**: se c'e' un soggetto chiaro (una persona, un oggetto) lo sfondo viene
   tolto; con paesaggi, palazzi o animali nell'erba la foto resta intera.
2. **Stile**: disegno nero su bianco "a china", come gli schemi trovati online:
   zone scure piene, linee per occhi, bocca, contorni.
3. **Larghezza**: tra 80 e 160 quadrati, piu' larga se la foto ha piu' dettagli.

Opzioni:

| Opzione | Effetto |
|---|---|
| `--detail 1` / `3` | meno / piu' dettaglio nello stile a china (default 2) |
| `--style colori` | pochi colori presi dalla foto invece del bianco e nero |
| `--colors N` | con `--style colori`: esattamente N colori |
| `--photo 120` | larghezza fissa in quadrati |
| `--remove-bg` / `--keep-bg` | toglie sempre / non toglie mai lo sfondo |

Funziona meglio con soggetto grande e sfondo semplice.

Per ogni immagine escono, accanto all'originale (o in `--out`):

| File | Contenuto |
|---|---|
| `<nome>_clean.pdf` | da stampare: un foglio A4 per pezzo; in alto a destra una miniatura del quadro intero col pezzo del foglio in rosso |
| `<nome>_clean.png` | lo schema intero in un'immagine |
| `<nome>_clean.svg` | solo con `--svg`: vettoriale, si apre e modifica in Figma |

Nel terminale viene stampato cosa e' stato rilevato, per esempio:

```
test.jpeg: 675x1200px -> 117 x 208 quadrati, lato 5.786 x 5.791px, ...
  griglia gia' disegnata: tengo righe 28-168, colonne 1-117
  schema 117 x 141 quadrati, 2 colori
```

## Schema stampato

- Linea spessa ogni 10 quadrati, numeri di righe e colonne sul bordo
  (numerazione globale: sul foglio 2 le colonne partono da 31).
- I fogli si interrompono sulle linee spesse, cosi' si affiancano facilmente.
- Sopra i quadrati scuri le linee diventano chiare, per restare visibili.
- Formato e dimensione: `--page a3`, `--cell-mm 4` (lato del quadrato stampato,
  default 3 mm). Il programma sceglie da solo verticale/orizzontale per usare meno fogli.

## Se qualcosa non torna

| Problema | Soluzione |
|---|---|
| Griglia sbagliata | guarda dove la mette: `--mode overlay` disegna le linee sopra l'originale |
| Lato quadrato sbagliato | imponilo: `--cell 12.5` (pixel) |
| Immagine ritagliata esattamente sul disegno | `--grid 40x30` (colonne x righe) |
| Troppi colori (sfumature da JPEG) | `--colors 8`, oppure `--merge 35` |
| Sfumature vicine unite per errore | `--merge 10` |
| Bordo di sfondo da togliere | `--trim` |
| Ritaglio automatico indesiderato | `--no-crop` |
| Quadrati non quadrati (immagine stirata) | `--nonsquare` |

Tutte le opzioni: `python gridify.py --help`.

## Come funziona

1. Per ogni confine tra pixel misura il salto di colore medio, in orizzontale e in verticale.
2. Cerca il passo che mette in fila piu' salti possibile (somma di fasori, come una
   trasformata di Fourier ma su passi non interi); tra multipli equivalenti sceglie il piu' grande.
3. La fase della stessa somma dice dove cade il primo bordo.
4. Il colore di ogni quadrato e' la mediana della sua parte centrale (ignora sfocatura e linee).
   Se l'immagine e' una foto di perline o diamond painting (quadrati con riflessi e fughe
   scure) usa tutto il quadrato, per non alternare perlina e fuga. Forzabile con `--inner`.
   Gli schemi di perline a mattoncino (file sfalsate di mezza perlina) non hanno una griglia
   quadrata: vengono ricampionati su quadrati, e l'affidabilita' stampata e' bassa.
5. I colori vengono raggruppati con k-means, cosi' il rumore JPEG non crea tonalita' finte.

## Test

```
python tests/test_gridify.py
```

Genera pixel art finta, la rovina (scala non intera, sfocatura, JPEG, margini,
griglia disegnata) e controlla lato, posizione, colori e file in uscita.
