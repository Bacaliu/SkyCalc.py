# SkyCalc.py
Achtung: Das Skript ist noch nicht fertig. Es kann sich jederzeit die Nuterschnittstelle und der Umfang ändern.

SkyCalc entstand aus dem Mangel einer guten Alternative von Calsky als jenes seinen Dienst einstellte. Daher habe ich mich optisch auch sehr stark daran orientiert. Die Berechnungen werden durch Pythons Skyfield modul durchgeführt.

![](screenshot.png)

Die rosa färbung kommt von meiner Browser-Konfiguration. Das ist eine einfache Tabelle ohne farbigen Hintergrund.
## Optional:
```bash
chmod +x skycalc.py
```
Dadurch wird die Datei ausführbar und ```./skycalc.py``` startet das Skript. Anderenfalls muss es jedesmal mit ```python skycalc.py``` gestartet werden was schlicht unpraktischer ist.

## Konfiguration
Oben im Dokument werden einige Variablen gesetzt. Die Kommentare beschreiben wie sie geändert werden ekönnen.

### Variable PATH
Das Skript muss wissen, wo es sich befindet. Bitte ohne abschließenden Slash

### Variable BROWSER
Das definiert welcher Browser zum öffnen der Tabelle verwendet wird. Natürlich kann man sich auch einfach ein Lesezeichen anlegen und dieses nach dem Durchlauf des Skriptes öffnen

### Variable TZ
Das definiert die Zeitzone, in der man lebt. Im Timedelta wird die differenz zur UTC angegeben, bei Mitteleuropäischer Sommerzeit sind es z.B. 2 Stunden also steht dort [...]hours = 2[...]

## Nutzerschnittstelle
Ohne Parameter erzeugt es die Tabelle in ```table.html``` im gleichen Ordner in dem es sich befindet.

### Parameter -open
``skycalc.py -open`` öffnet die Tabelle nach dem Erstellen im angegebenen Browser.

### Parameter -dur [int]
DURation also Dauer gibt die Anzahl der Stunden an, bis zu der in Zukunft gerechnet wird. Standartmäßig sind es 24.

### Parameter -sat
Das aktiviert die Berechnung und Anzeige von Satellitenüberflügen. Standartmäßig ist es deaktiviert

### Parameter -sat-mag [float]
Das aktiviert die Berechnung und Anzeige von Satellitenüberflügen UND setzt die Helligkeit nach der gefiltert wird auf [float] magnituden. Satelliten, die während des Überfluges immer dunkler sind werden also nicht geplottet und nicht dargestellt

