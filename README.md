# autograd

Eine handgeschriebene Autograd-Engine in Go (im Stil von micrograd) und darauf aufbauende MLPs,
angewendet auf Tic-Tac-Toe (bzw. das m,n,k-Spiel, bisher nur 3x3).

- `nn/`: Autograd-Engine (`Value`, siehe [nn/value_readme.md](nn/value_readme.md)) und MLP.
- `tictactoe/`: Spiellogik, Minimax, Erzeugen und Annotieren der Stellungen, D4-Symmetrien.
- `scoremodel/`: das Scoremodel (MLP, Losses, Training, Evaluation, KI-Spieler).
- `cmd/`: die Programme `datagen`, `train`, `tictactoe` und `xor`.

## Schnellstart

Daten erzeugen (schreibt `datasets/boards_3x3_perfect.ndjson`):

```
cmd/datagen/generate_boards_3x3_perfect.sh
```

Optional die Modelle neu trainieren (die trainierten Modelle liegen bereits in `models/`):

```
cmd/train/train_all.sh
```

Gegen Minimax oder eines der Modelle spielen:

```
go run ./cmd/tictactoe
```

## Datenformat

Der Datensatz ist eine NDJSON-Datei mit einer Stellung pro Zeile. Er enthaelt alle erreichbaren
Stellungen mit mindestens zwei legalen Zuegen (keine Endstellungen, keine erzwungenen Zuege):

```
{"Board":[0,0,0,0,0,0,0,0,2],"Player":1,"Scores":[{"Pos":4,"Score":0},{"Pos":0,"Score":-0.531441},...]}
```

- `Board`: die 9 Felder zeilenweise (Index `r*3+c`), 0 = leer, 1 = am Zug, 2 = Gegner.
- `Player`: wer am Zug ist.
- `Scores`: fuer jeden legalen Zug (`Pos`) der Minimax-Wert aus Sicht des Spielers am Zug.

Der Score ist ein diskontierter Minimax-Wert in [-1, 1]: ein Gewinn nach `d` Halbzuegen zaehlt
`0.9^d`, eine Niederlage `-0.9^d`, ein Remis 0. Schnelle Gewinne und spaete Niederlagen werden so
bevorzugt. Die Modelle in `models/` werden allerdings nur auf das Vorzeichen trainiert, also auf die
Werte 1 (Gewinn), 0 (Remis) und -1 (Niederlage), siehe `-outcome-classes` unten.

## Kanonisch vs. absolut

- **Absolut**: X = 1 und O = 2, egal wer am Zug ist. So speichert das laufende Spiel (`Game`) das
  Board.
- **Kanonisch** (aus Sicht des Spielers am Zug): 1 = eigene Steine, 2 = Steine des Gegners.

Ist X am Zug, sind beide Kodierungen gleich:

```
X . O        absolut:   [1,0,2, 0,0,0, 0,0,0]
. . .        kanonisch: [1,0,2, 0,0,0, 0,0,0]    (X = 1 = eigene Steine)
. . .
```

Ist O am Zug, tauschen 1 und 2 die Rollen:

```
X . O        absolut:   [1,0,2, 0,1,0, 0,0,0]
. X .        kanonisch: [2,0,1, 0,2,0, 0,0,0]    (O = 1 = eigene Steine, X = 2 = Gegner)
. . .
```

Das Modell arbeiten verwendet kanonische Daten. Dadurch muss es nur lernen, "fuer den Spieler am 
Zug" zu spielen, nicht separat fuer X und O. Im Spiel wird das absolute Board vor der Auswertung 
umkodiert.

Das Netz muss jedes Muster (z.B. "zwei eigene in einer Reihe, hier gewinnen") nur einmal lernen 
statt je einmal fuer X und O. Das spart Kapazitaet und jedes Muster bekommt doppelt so viele 
Trainingsbeispiele.

## Scoremodel

Das Scoremodel ist ein MLP, das fuer jedes Feld einen Score vorhersagt. Gespielt wird der legale Zug
mit dem hoechsten Score.

```
Board (kanonisch)      Eingabe, 18 Werte                      Ausgabe, 9 Werte
X . O                  Ebene 0 "eigene"    1 0 0  0 1 0  0 0 0     s0 s1 s2
. X .        ->        (Indizes 0-8)                          ->   s3 s4 s5
. . O                  Ebene 1 "Gegner"    0 0 1  0 0 0  0 0 1     s6 s7 s8
                       (Indizes 9-17)
```

- Eingabe: zwei One-Hot-Ebenen mit je 9 Werten, eine fuer die eigenen Steine und eine fuer die des
  Gegners. Mit nur einem Wert pro Feld (0, 1, 2) waere ein gegnerischer Stein fuer jedes Neuron
  "doppelt so viel" wie ein eigener. Mit One-Hot hat jede Kombination aus Feld und Stein ein eigenes
  Gewicht.
- Ausgabe: 9 Scores, einer pro Feld. Besetzte Felder werden beim Training und beim Spielen maskiert.
- Architektur der Modelle in `models/`: `18-32-32-9`, ReLU in den Hidden Layers, tanh am Ausgang.
- Mit `-outcome-classes` wird nicht auf den diskontierten Score trainiert, sondern nur auf dessen
  Vorzeichen (1 = Gewinn, 0 = Remis, -1 = Niederlage).

## Loss-Funktionen

Alle Losses werden nur ueber die legalen Zuege berechnet (`-loss`):

- **mse**: mittlerer quadratischer Fehler zwischen vorhergesagten und Ziel-Scores. Das Modell lernt
  die Werte selbst, nicht nur die Reihenfolge der Zuege.
- **hinge**: paarweiser Ranking-Loss. Fuer jedes Paar (bester Zug, schlechterer Zug) kostet es
  `max(0, margin + s_schlechter - s_bester)`, gemittelt ueber alle Paare. Es zaehlt nur, dass die
  besten Zuege mit Abstand `margin` vorne liegen.
- **crossentropy**: Softmax ueber die legalen Zuege, Ziel ist eine Gleichverteilung ueber die besten
  Zuege. Das Modell lernt also eine Zugwahrscheinlichkeit statt eines Werts.

## Training

Trainiert wird mit `cmd/train`, die Einstellungen der Modelle in `models/` stehen in den Scripts
`cmd/train/scoremodel_3x3_*.sh`. Ein Lauf hat vier Schritte: Daten aufteilen, Epochen trainieren,
regelmaessig Matchups spielen und am Ende auswerten und speichern. Fuer Reproduzierbarkeit wird ein
fester Seed fuer alles (Aufteilung, Startgewichte, Reihenfolge der Stellungen, Minimax-Gegner) 
verwendet.

### Aufteilung und D4-Symmetrien

Mit `-test-fraction` (Standard 0.1) wird ein Teil der Stellungen aus dem Training herausgenommen.
Diese Stellungen sieht das Modell beim Training nie, sie werden erst danach ausgewertet. Dabei
spielen die Symmetrien des Boards eine Rolle.

Das 3x3-Board hat 8 Symmetrien (Identitaet, 3 Drehungen, 4 Spiegelungen), die Diedergruppe D4. Eine
gedrehte oder gespiegelte Stellung hat dieselben Scores, nur an entsprechend verschobenen Feldern.

Die 8 Varianten einer Stellung (Drehungen im Uhrzeigersinn, Spiegelungen an der gezeigten Achse):

```
Identitaet    Drehung 90    Drehung 180   Drehung 270
X O .         . . X         . . .         . . .
. . .         . . O         . . .         O . .
. . .         . . .         . O X         X . .

Spiegelung |  Spiegelung -  Spiegelung \  Spiegelung /
. O X         . . .         X . .         . . .
. . .         . . .         O . .         . . O
. . .         X O .         . . .         . . X
```

Der Zug auf Feld 2 (rechts neben O) im Original entspricht nach der Drehung um 90 Grad dem Zug auf
Feld 8 (unter O) und hat denselben Score.

Der Datensatz enthaelt bereits alle symmetrischen Varianten jeder Stellung, das Training augmentiert
deshalb nicht zusaetzlich. Beim Aufteilen in Trainings- und Testdaten (`D4Split`) landen alle
Varianten derselben Stellung immer in derselben Menge, damit das Modell im Test keine Stellungen
sieht, die es als gedrehte oder gespiegelte Variante schon aus dem Training kennt.

### Epochen

Eine Epoche ist ein Durchlauf ueber alle Trainingsstellungen, in jeder Epoche neu gemischt. Nach
jeder Stellung werden die Parameter sofort angepasst (Online-SGD, ein Update pro Stellung):

```
w = w - learning_rate * (grad + 2 * weight_decay * w)     (Gewichte)
b = b - learning_rate * grad                              (Biases)
```

- `-learning-rate`: die Schrittweite.
- `-weight-decay`: zieht die Gewichte leicht gegen 0 (L2-Regularisierung, gegen Overfitting). Biases
  sind ausgenommen, sie verschieben die Funktion nur und machen sie nicht komplexer.
- `-epochs`: die Hoechstzahl der Epochen.

### Matchups

Der Loss ist nur ein Stellvertreter, entscheidend ist, wie gut das Modell spielt. Deshalb spielt das
Modell alle `-matchup-interval` Epochen und nach der letzten Epoche ein Matchup: `-matchup-games`
Partien gegen den Gegner `-matchup-opponent`, abwechselnd als X und als O. Der Score ist
`(Siege + 0.5 * Remis) / Partien`. Das bisher beste Modell spielt dasselbe Matchup, am Anfang ist
das das untrainierte Startmodell. Dann wird entschieden:

1. Beim ersten Matchup wird das Modell immer behalten. So geht das erste Intervall Training nie
   zugunsten des untrainierten Startmodells verloren.
2. Danach wird das Modell behalten und neues bestes Modell, wenn sein Score hoeher ist als der des
   besten Modells.
3. Bei gleichem Score entscheidet der Loss: Das Modell wird behalten, wenn der mittlere Loss seiner
   letzten Epoche niedriger ist als der des besten Modells. Als Verbesserung zaehlt das aber nicht,
   siehe `-matchup-patience` unten.
4. Sonst wird das Modell auf das beste Modell zurueckgesetzt und das Training geht von dort weiter.
   Weil die Stellungen jede Epoche neu gemischt werden, nimmt es dabei einen anderen Weg.

Das Training endet nach `-epochs` Epochen oder wenn sich der Score `-matchup-patience` Matchups in
Folge nicht verbessert hat. Ein Modell, das nur wegen des niedrigeren Loss behalten wurde, zaehlt
dabei nicht als Verbesserung. Das Ergebnis ist das beste Modell. Ohne Matchups
(`-matchup-interval 0`) ist es das Modell nach der letzten Epoche.

Als Gegner gibt es:

- **minimax**: perfekter Minimax. Unter gleich guten Zuegen waehlt er zufaellig, aber jedes Matchup
  beginnt mit demselben Seed. Dasselbe Modell bekommt also immer dieselben Partien und denselben
  Score, das Modell und das beste Modell werden fair verglichen. Gegen perfekten Minimax kann man
  nicht gewinnen, der beste moegliche Score ist 0.5 (nur Remis).
- **self [WIP]**: das bisher beste Modell. Wuerde jedes Modell in derselben Stellung immer denselben 
  Zug waehlen, gaebe es nur zwei verschiedene Partien (eine pro Seite) und nur die Scores 0, 0.25, 
  0.5, 0.75 und 1. Die zufaellige Zugwahl (siehe unten) bringt mehr verschiedene Partien, aber nur
  in Stellungen, in denen das Modell mehrere Zuege fast gleich bewertet.

In den Matchups spielen das Modell und das beste Modell (bei **self** also auch der Gegner) nicht
immer ihren besten Zug, sondern zufaellig einen ihrer bis zu 3 besten Zuege, deren Score hoechstens
in einem gewissen Fenster (momentan auf 0.1 hardgecoded) unter dem besten liegt. Einen klar am 
besten bewerteten Zug spielen sie also immer, Zufall gibt es nur unter fast gleich bewerteten 
Zuegen.

TODO: Bei verschienen Losses liegen die Scores auf einer anderen Skala (insbesonder bei 
crossentropy in Vergleich zu der anderen beiden). D.h. dieses Fenster waere ein weiterer 
Hyperparameterkandidat (aber langsam reichts).

### Nach dem Training

Das Ergebnis wird ausgewertet:

- Spielbaum auf Trainings- und Testdaten: wie oft der gewaehlte Zug das optimale Ergebnis (Gewinn,
  Remis oder Niederlage) haelt.
- Perfektion: dasselbe ueber alle erreichbaren Stellungen.
- 1000 Partien gegen Minimax (hardgecoded), damit die Ergebnisse verschiedener Modelle vergleichbar
  bleiben.

Mit `-out` werden die Gewichte als JSON gespeichert, im Feld `info` zusammen mit allen Einstellungen
und Ergebnissen. Daneben landen die Trainings- und Testdaten als `<name>_train.ndjson` und
`<name>_test.ndjson`.
