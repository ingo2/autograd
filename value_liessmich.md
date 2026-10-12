# value.go - Erklärung an einem Beispiel

`Value` ist eine einfache **Autograd-Engine**: Sie stellt eine Funktion intern als Graph dar und
berechnet automatisch die Gradienten aller Knoten via Backpropagation.

## Beispielfunktion

Sei `L(a, b, c) = 2ab^2 + abc = (ab) * (2b + c)`

L kann an einem beliebigen Punkt ausgewertet werden, z.B. `a=1, b=-2, c=3`:

```
L(1, -2, 3) = (1 * (-2)) * (2 * (-2) + 3) = (-2) * (-1) = 2
```

Die partiellen Ableitungen:

```
dL/da = 2b^2 + bc = 2(-2)^2  + (-2)*3 =  8 - 6 =  2
dL/db = 4ab  + ac = 4*1*(-2) + 1*3    = -8 + 3 = -5
dL/dc = ab        = 1*(-2)            = -2
```

## Backpropagation

### Motivation

Bei neuronalen Netzen wird der Gradient der Verlustfunktion L bzgl. aller Gewichte benötigt:

```
grad L = (dL/dw1, ..., dL/dwn)
```

Symbolisch (z.B. per Computeralgebra) erhält man zwar exakte Ausdrücke, diese wachsen bei tief
verschachtelten Funktionen aber stark an, da gemeinsame Teilausdrücke vielfach dupliziert werden.
Numerisch ließe sich jede partielle Ableitung über einen Differenzenquotienten
annähern:

```
dL/dwi ~ [L(w1, ..., wi + h, ..., wn) - L(w1, ..., wi - h, ..., wn)] / (2h)
```

Das erfordert 2n Auswertungen von L (mit dem einseitigen Differenzenquotienten
`[L(..., wi + h, ...) - L(w1, ..., wn)] / h` immerhin noch n+1), und jede Auswertung kostet selbst
etwa O(n) Operationen.
Insgesamt also O(n^2), bei möglicherweise Milliarden von Gewichten!

Ein effizientes Verfahren ist **Backpropagation**: Es berechnet den exakten Gradienten bzgl. aller
Gewichte mit einem Vorwärts- und einem Rückwärtsdurchlauf. Der Rückwärtsdurchlauf kostet etwa
doppelt so viel wie der Vorwärtsdurchlauf, das entspricht insgesamt etwa 3 Auswertungen von L, 
unabhängig von n. Gegenüber dem Differenzenquotienten ergibt sich ein Speedup von etwa n/3.

Beispiel: Angenomment ein Modell mit 10^9 Gewichten braucht 1 Sekunde für eine Auswertung von L. Mit
Backpropagation dauert ein vollständiger Gradient etwa 3 Sekunden. Mit dem Differenzenquotienten
wären es etwa 10^9 Sekunden, also rund 30 Jahre, für einen einzigen Trainingsschritt.

Dafür braucht Backpropagation mehr Speicher. Der Rückwärtsdurchlauf benötigt die Werte, die im
Vorwärtsdurchlauf an jedem Knoten berechnet wurden (im Beispiel unten etwa `dL/dd = e`). Diese
Werte müssen daher bis zum Rückwärtsdurchlauf gespeichert bleiben.

### Zerlegung in einen Graphen

Wir zerlegen L mittels Hilfsvariablen in Knoten.
Ersetze in `L = (ab) * (2b + c)` die Terme: `d = a*b`, `e = 2b + c`, dann ist `L = d * e`.
`e` lässt sich weiter zerlegen, ist hier aber nicht nötig. An jedem Knoten muss lediglich die
lokale Ableitung bekannt sein. Wie fein man den Graphen aufteilt, ist ein Kompromiss. Feine Knoten 
sind flexibler, da nur wenige Grundoperationen eine eigene Ableitung brauchen. Grobe Knoten (z.B. 
`tanh` als ein Knoten statt aus `exp`, `+` und `/` zusammengesetzt) sparen Rechenzeit und Speicher, 
da weniger Zwischenwerte anfallen, ihre lokale Ableitung muss aber von Hand implementiert werden.

Es ergibt sich folgender Graph:

```
a = 1
 \
  [d = a*b]
 /          \
b = -2       [L = d*e]
 \          /
  [e = 2b+c]
 /
c = 3
```

Ziel: an jedem Knoten Funktionswert und Gradient bestimmen.

### Forward Pass

```
d = a * b      =  1 * (-2)  = -2
e = 2*(-2) + 3 = -4 + 3     = -1
L = d * e      = -2 * (-1)  =  2
```

### Backward Pass (Kettenregel)

```
dL/dL = 1  (per Definition)

dL/dd = dL/dL * d(d*e)/dd = 1 * e = -1
dL/de = dL/dL * d(d*e)/de = 1 * d = -2

dL/da = dL/dd * d(a*b)/da  = -1 * b = -1 * (-2) =  2
dL/dc = dL/de * d(2b+c)/dc = -2 * 1 =             -2
```

`b` hat 2 Elternknoten (`d` und `e`) => totale Ableitung als Summe beider Beiträge:

```
dL/db = dL/dd * d(a*b)/db + dL/de * d(2b+c)/db
      = -1 * a            + (-2)  * 2
      = -1 + (-4) = -5
```

Diese Werte stimmen mit den direkt berechneten partiellen Ableitungen überein.

## Verwendung von value.go

`value.go` baut genau einen solchen Graphen auf, wenn man eine Berechnung ausführt:

```go
a := NewValue(1)
b := NewValue(-2)
c := NewValue(3)

d := a.Mul(b)
e := b.Mul(NewValue(2)).Add(c)
L := d.Mul(e)

L.Backward() // Berechnet alle Gradienten.
```

Danach enthält `a.Grad = 2`, `b.Grad = -5`, `c.Grad = -2`.
