#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta, timezone
from math import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pytz
import requests
import skyfield
from skyfield.api import N, W, E, S, wgs84, load, utc, Star
from skyfield.framelib import ecliptic_frame
from skyfield.data import hipparcos, stellarium
from skyfield.magnitudelib import planetary_magnitude
from skyfield.trigonometry import position_angle_of
from skyfield.units import Angle
from skyfield import almanac, searchlib
import sys
from tqdm.auto import tqdm
## Konstanten - Nicht bearbeiten | Constants - don't change
ts = load.timescale()
SPACE = "&nbsp;"
EPH = load('de421.bsp')
## Einstellungen - bei Bedarf bearbeiten | Settings - set them if needed
# PATH = Pfadt zu dieser Datei | PATH = path to this file
PATH = "/home/adrian/Skripte/astro"
# Anzahl der Stunden zwischen UTC und unserer Zeitzone | number of hours between UTC and our Timezone
TZ = timezone(timedelta(hours = 2))

## Herunterladen der Satellitendaten | Downloading satellite-data
VISUAL = load.tle_file('https://celestrak.com/NORAD/elements/visual.txt', reload = False)
STATIONS = load.tle_file("https://celestrak.com/NORAD/elements/stations.txt", reload = False)
STARLINK = load.tle_file("https://celestrak.com/NORAD/elements/starlink.txt", reload = False)

def position(lon:float, lat:float):
    ortE = EPH["EARTH"]+wgs84.latlon(lat*N, lon*E)
    ort = wgs84.latlon(lat, lon)
    return ort, ortE

def ptime(dt:datetime)->str:
    return dt.astimezone(TZ).strftime("%Hh%Mm%Ss")

def runde(zahl:float, n:int, m:int=0)->str:
    # rundet die Zahl zahl auf n stellen und füllt mit Leerzeichen auf m Stellen auf.
    # round the number zahl to n digits and fills with spaces to at least m digits.
    if n == 0:
        ret = str(int(zahl))
    else:
        ret = str(round(zahl, n))
    ret = max(m-len(ret), 0)*SPACE + ret
    return ret

def mag(ts0, sat, lon:float, lat:float, eph = EPH):
    offset = offset_mag(ts0, sat, lon, lat)
    tar_n = sat.target_name.split("#")[1]
    satid = sat_id(sat)
    ## ist schon im cache?
    if os.path.exists(f"{PATH}/satmag.txt"):
        with open(f"{PATH}/satmag.txt", "r") as f:
            for l in f:
                if l.split(":")[0] == satid:
                    m = l.split(":")[1]
                    try:
                        return float(m)+offset
                    except:
                        return "?"
        ## nicht im cache.
    url = f"http://heavens-above.com/satinfo.aspx?satid={satid}"
    resp = requests.get(url)
    try:
        r = resp.text.split("(Magnitude)")[1][10:].split(" ")[0]
        m = float(r)
    except:
        m = "?"
    with open(f"{PATH}/satmag.txt", "a") as f:
        f.write(f"{satid}:{m}\n")
    try:
        return m+offset
    except:
        return "?"

def offset_mag(ts0, satellite, lon:float, lat:float, eph = EPH)->float:
    ort, ortE = position(lon, lat)
    g = (EPH["EARTH"]+satellite).at(ts0).observe(EPH["SUN"]).separation_from((EPH["EARTH"]+satellite).at(ts0).observe(ortE))
    d = (satellite-ort).at(ts0).altaz()[2]
    return - 15 + 5*log10(d.km) -2.5*log10(sin(g.radians) + (pi-g.radians)*cos(g.radians))

def rich(alt:float)->str:
    for d, n in zip(np.arange(0, 360, 22.5), ["N", "NNO", "NO", "ONO",
                                        "O", "OSO", "SO", "OSO",
                                        "S", "SSW", "SW", "WSW", 
                                        "W", "WNW", "NW", "NNW"]):
        if abs((alt%360)-d) < 11.25:
            return n+(SPACE*(3-len(n)))
    return f"N{SPACE*2}"

def mix(n:float, l:float, u:float)->float:
    return max(l, min(u, n))

def satcol(ts0, satellite, lon:float, lat:float)->tuple:
    ort, ortE = position(lon, lat)

    if not satellite.at(ts0).is_sunlit(EPH):
        return (0, 0, 0.25)

    m = mag(ts0, satellite, lon, lat)
    try:
        m = float(m)
    except:
        return (0.5, 0.5, 0.5)

    return (mix(m, 0, 6)/6, mix(6-m, 0, 6)/6, mix(-m, 0, 4)/4)

####
# DARSTELLUNG
# 
####

def html_row(a:str, b:str, c:str, params:str="")->str:
    return f'<tr{params}><td style="text-align:center">{a}</td><td style="text-align:center;">{b}</td><td>{c}</td></tr>'

def big_emoji(emoji:str)->str:
    return f'<span style = "font-size:200%">{emoji}</span><br>'

def mondphase(ts0)->float:
    _, slon, _ = EPH["EARTH"].at(ts0).observe(EPH["SUN"]).apparent().frame_latlon(ecliptic_frame)
    _, mlon, _ = EPH["EARTH"].at(ts0).observe(EPH["MOON"]).apparent().frame_latlon(ecliptic_frame)
    phase = (mlon.degrees - slon.degrees) % 360.0
    return phase

def mond_emoji(phase:float)->str:
    for s, p in zip(["&#127761;", "&#127762;", "&#127763;",
        "&#127764;", "&#127765;", "&#127766;", "&#127767;", "&#127768;"],
        range(0, 360, 45)):
        if abs((phase-p)) < 22.5:
             return big_emoji(s)

def mond_url()->str:
    return "https://theskylive.com/moon-info"

def mond_darstell(ts)->str:
    return f'{mond_emoji(mondphase(ts))}<a href="{mond_url()}" target = "_blank">Mond</a>'

def sat_id(sat)->str:
    tar_n = sat.target_name.split("#")[1]
    satid = ""
    for s in tar_n:
        if s == " ":
            break
        satid+=s
    return satid

def sat_url(sat)->str:
    return f"http://heavens-above.com/satinfo.aspx?satid={sat_id(sat)}"

def sat_emoji()->str:
    return big_emoji("&#128752;")


def sat_darstell(sat)->str:
    return f'{sat_emoji()}<a href="{sat_url(sat)}" target="_blank">{sat.name}</a>'

def planet_url(name:str)->str:
    return f"https://theskylive.com/{name.split('_')[0].lower()}-info"

def planet_emoji(name:str)->str:
    s = {"mercury":"&#x263F;",
         "venus":"&#x2640;",
         "mars":"&#x2642;",
         "jupiter":"&#x2643",
         "saturn":"&#x2644",
         "uranus":"&#x26E2",
         "neptune":"&#x2646"}
    return big_emoji(s[name.split('_')[0].lower()])

def planet_darstell(name:str)->str:
    return f'{planet_emoji(name)}<a href ="{planet_url(name.lower())}" target="_blank">{name.split("_")[0].capitalize()}</a>'

def sonne_emoji(helligkeit:int)->str:
    loe = ["&#x1F30C;", "&#x1F303;", "&#x1F306;", "&#x1F307;", "&#x2600;&#xFE0F;"]
    return big_emoji(loe[helligkeit])

def sonne_darstell(helligkeit:int)->str:
    if helligkeit == 3:
        name = "Sonne"
    else:
        name = "Dämmerung"
    return f'{sonne_emoji(helligkeit)}<a href="{planet_url("sun")}" target="_blank">{name}'

def AltAzRaDecDis(eph, ts0, lon, lat, elev):
    ort, ortE = position(lon, lat)
    geographic = wgs84.latlon(lat, lon, elev)
    alt, az, dis = ortE.at(ts0).observe(eph).apparent().altaz()
    ra, dec, _ = geographic.at(ts0).from_altaz(alt, az).radec()

    return alt, az, ra, dec, dis

######################################################################################
## SUCHE
######################################################################################

def satellite_events(satellites, ts0, ts1,
        min_degrees:float, lon:float, lat:float, sun_deg:float=-9, max_mag:float=4):

    ort, ortE = position(lon, lat)
    events = []
    index = 0
    for sat in tqdm(satellites, desc = "suche Satelliten"):
        t, event = sat.find_events(ort, ts0, ts1, altitude_degrees=0)
        if len(event) < 3:
            continue
        while event[0] != 0:
            t = t[1:]
            event = event[1:]
        if len(event) < 3:
            continue
        while event[-1] != 2:
            t = t[:-1]
            event = event[:-1]
        for i in range(0, len(event), 3):
            if (sat-ort).at(t[1+i]).altaz()[0].degrees < min_degrees:
                continue
            e = dict()
            e["diff"] = (sat-ort)
            e["satellite"] = sat
            e["index"] = index

            if ortE.at(t[i+1]).observe(EPH["SUN"]).apparent().altaz()[0].degrees > sun_deg:
                continue

            for j, name in enumerate(["Aufgang", "Kulmination", "Untergang"]):
                e[name] = dict()
                e[name]["ts"] = t[i+j]
                e[name]["dt"] = t[i+j].utc_datetime()
                e[name]["mag"] = mag(e[name]["ts"], e["satellite"], lon, lat)
                e[name]["altaz"] = e["diff"].at(e[name]["ts"]).altaz()
                e[name]["Distanz"] = e["diff"].at(e[name]["ts"]).distance()
                e[name]["Geschwindigkeit"] = e["diff"].at(e[name]["ts"]).speed()
                e[name]["Name"] = name

            for name, test,  delta in zip(["Erscheint", "Verschwindet"],
                                          ["Aufgang", "Untergang"],
                                          [timedelta(seconds=1), timedelta(seconds=-1)]):
                if not e["diff"].at(e[test]["ts"]).is_sunlit(EPH):
                    ti = e[test]["ts"].utc_datetime()
                    while not e["diff"].at(ts.from_datetime(ti)).is_sunlit(EPH):
                        ti += delta*10
                    while e["diff"].at(ts.from_datetime(ti)).is_sunlit(EPH):
                        ti -= delta
                    e[name] = dict()
                    e[name]["dt"] = ti
                    e[name]["ts"] = ts.from_datetime(ti)
                    try:
                        e[name]["mag"] = float(mag(e[name]["ts"], e["satellite"], lon, lat))
                    except:
                        e[name]["mag"] = None
                    e[name]["altaz"] = e["diff"].at(e[name]["ts"]).altaz()
                    e[name]["mag"] = mag(e[name]["ts"], e["satellite"], lon, lat)
                    e[name]["Distanz"] = e["diff"].at(e[name]["ts"]).distance()
                    e[name]["Geschwindigkeit"] = e["diff"].at(e[name]["ts"]).speed()
                    e[name]["Name"] = name

            if min([(e[s]["mag"] if isinstance(e[s]["mag"], float) else 99 ) if isinstance(e[s], dict) else 99 for s in e]) < max_mag:
                events.append(e)
                index += 1
    return events

def mond_events(ts0, ts1, lon:float, lat:float, elev:float):
    def altF(ts):
        alt, az, ra, dec, dis = AltAzRaDecDis(EPH["MOON"], ts, lon, lat, elev)
        return alt.degrees
    altF.step_days = 0.5
    
    ort, ortE = position(lon, lat)

    ret = []
    t, y = almanac.find_discrete(ts0, ts1, almanac.moon_phases(EPH))
    ## PHASEN
    for ti, yi in zip(t, y):
        alt, az, ra, dec, dis = AltAzRaDecDis(EPH["MOON"], ti, lon, lat, elev)
        this = dict()
        this["dt"] = ti.utc_datetime()
        this["html"] = html_row(
            ptime(ti.utc_datetime()),
            mond_darstell(ti),
            f"{almanac.MOON_PHASES[yi]}<br>{alt}{SPACE*2}{az}")
        ret.append(this)
    
    t, y = almanac.find_discrete(ts0, ts1, almanac.moon_nodes(EPH))
    ## NODES
    for ti, yi in zip(t, y):
        alt, az, ra, dec, dis = AltAzRaDecDis(EPH["MOON"], ti, lon, lat, elev)
        this = dict()
        this["dt"] = ti.utc_datetime()
        this["html"] = html_row(ptime(ti.utc_datetime()),
                                mond_darstell(ti),
                                f"{almanac.MOON_NODES[yi]}{SPACE*2}az:{SPACE}{runde(az.degrees, 0, 3)}º{SPACE}{rich(az.degrees)}{SPACE*2}h:{SPACE}{runde(alt.degrees, 0, 3)}º<br>Phase:{SPACE}{runde(mondphase(ti), 1, 5)}º<br>RA:{SPACE}{ra}{SPACE*2}DEC:{SPACE}{dec}")
        ret.append(this)

    t, y = almanac.find_discrete(ts0, ts1, almanac.risings_and_settings(EPH, EPH['MOON'], ort))
    ## AUF UNTER
    for ti, yi in zip(t, y):
        alt, az, ra, dec, dis = AltAzRaDecDis(EPH["MOON"], ti, lon, lat, elev)
        this = dict()
        this["dt"] = ti.utc_datetime()
        this["html"] = html_row(ptime(ti.utc_datetime()),
                       mond_darstell(ti), f"{'Aufgang'+SPACE*4 if yi else 'Untergang'+SPACE*2}{SPACE*2}az:{SPACE}{runde(az.degrees, 1, 5)}º{SPACE}{rich(az.degrees)}{SPACE*2}<br>Phase:{SPACE}{runde(mondphase(ti), 1, 5)}º<br>RA: {ra}{SPACE*2}DEC: {dec}{SPACE*2}")
        ret.append(this)


    t, y = searchlib.find_maxima(ts0, ts1, altF)
    ## Kulmination
    for ti, yi in zip(t, y):
        alt, az, ra, dec, dis = AltAzRaDecDis(EPH["MOON"], ti, lon, lat, elev)
        if alt.degrees < 0: continue
        this = dict()
        this["dt"] = ti.utc_datetime()
        this["html"] = html_row(ptime(ti.utc_datetime()),
                       mond_darstell(ti),
                       f"<b>Kulmination</b>{SPACE*2}az:{SPACE}{runde(az.degrees, 0, 3)}º{SPACE}{rich(az.degrees)}{SPACE*2}<b>h:{SPACE}{runde(alt.degrees, 0, 3)}º</b><br>RA:{SPACE}{ra}{SPACE*2}DEC:{SPACE}{dec}{SPACE*2}Phase:{SPACE}{runde(mondphase(ti), 1, 5)}º")
        ret.append(this)

    return ret

def planeten_events(ts0, ts1, lon:float, lat:float, elev:float):
    def altF(t)->float:
        return ortE.at(t).observe(EPH[planet]).apparent().altaz()[0].degrees
    altF.step_days = 0.5

    ort, ortE = position(lon, lat)
    ret = []

    for planet in ["MERCURY", "VENUS", "MARS", "JUPITER_BARYCENTER", "SATURN_BARYCENTER", "URANUS_BARYCENTER", "NEPTUNE_BARYCENTER"]:
        t, y = almanac.find_discrete(ts0, ts1, almanac.risings_and_settings(EPH, EPH[f'{planet}'], ort))
        ## AUF UNTER
        for ti, yi in zip(t, y):
            try:
                m = planetary_magnitude(ortE.at(ti).observe(EPH[f'{planet}']))
                m = "<b>"+str(runde(m, 1, 4))+" mag</b>"
            except:
                m = f"{SPACE*2}?{SPACE*2}mag"
            alt, az, ra, dec, dis = AltAzRaDecDis(EPH[planet], ti, lon, lat, elev)
            this = dict()
            this["dt"] = ti.utc_datetime()
            this["html"] = html_row(ptime(ti.utc_datetime()),
                        planet_darstell(planet),
                        f"{'Aufgang'+SPACE*4 if yi else 'Untergang'+SPACE*2}{SPACE*2}az: {runde(az.degrees, 1, 5)}º {rich(az.degrees)}<br>{m}<br>RA: {ra}{SPACE*2}DEC: {dec}{SPACE*2}")
            ret.append(this)
            
        t, y = searchlib.find_maxima(ts0, ts1, altF)
        ## Kulmination
        for ti, yi in zip(t, y):
            try:
                m = planetary_magnitude(ortE.at(ti).observe(EPH[f'{planet}']))
                m = "<b>"+str(runde(m, 1, 4))+" mag</b>"
            except:
                m = f"{SPACE*2}?{SPACE*2}mag"
            alt, az, ra, dec, dis = AltAzRaDecDis(EPH[planet], ti, lon, lat, elev)
            if alt.degrees < 0:
                continue
            this = dict()
            this["dt"] = ti.utc_datetime()
            this["html"] = html_row(ptime(ti.utc_datetime()),
                        planet_darstell(planet),
                        f"<b>Kulmination</b>{SPACE*2}az: {runde(az.degrees, 0, 3)}º {rich(az.degrees)}{SPACE*2}<b>h: {runde(alt.degrees, 0, 3)}º</b><br>{m}<br>RA: {ra}{SPACE*2}DEC: {dec}")
            ret.append(this)
    return ret

def sonne_events(ts0, ts1, lon:float, lat:float):
    dämmerungen = ["Nacht", "Astronomische Dämmerung", "Nautische Dämmerung", "Bürgerliche Dämmerung", "Tag"]
    dämbesch = [["", "Es ist maximal Dunkel", "Sternenbilder werden sichtbar", "helle Sterne und Planeten tauchen auf", ""], ["", "schwache Sterne verblassen", "Sternenbilder lösen sich auf", "helle Sterne und Planeten verblassen", ""]]
    ort, ortE = position(lon, lat)
    ret = []
    
    f = almanac.dark_twilight_day(EPH, ort)
    times, events = almanac.find_discrete(ts0, ts1, f)

    previous_e = f(ts0).item()
    for t, e in zip(times, events):
        pos = ortE.at(t).observe(EPH[f'SUN']).apparent().altaz()
        this = dict()
        this["dt"] = t.utc_datetime()
        this["html"] = "<tr>"
        if previous_e < e:
            if e >= 4:
                this["html"] = html_row(ptime(t.utc_datetime()),
                    f"{sonne_darstell(e-1)}",
                    f"<b>Aufgang</b>{SPACE*6}az: {round(pos[1].degrees)}º {rich(pos[1].degrees)}")
            else:
                this["html"] = html_row(ptime(t.utc_datetime()),
                    f"{sonne_darstell(e-1)}",
                    f"<b>{dämmerungen[e]}</b>{SPACE*2}{dämbesch[1][e]}")

        else:
            if e >= 3:
                this["html"] = html_row(ptime(t.utc_datetime()),
                f"{sonne_darstell(e)}",
                f"<b>Untergang</b>{SPACE*4}az: {round(pos[1].degrees)}º {rich(pos[1].degrees)}")
            else:
                this["html"] = html_row(ptime(t.utc_datetime()),
                f"{sonne_darstell(e)}",
                f"<b>{dämmerungen[e+1]}</b>{SPACE*2}{dämbesch[0][e+1]}")
        previous_e = e

        this["html"] += "</tr>"
        ret.append(this)
    return ret

def sat_events_to_html(events, lon:float, lat:float, sat_mag = 4):
    rows = list()
    ort, ortE = position(lon, lat)
    for i, ev in enumerate(events):
        parts = list()
        for k, e in ev.items():
            if isinstance(e, dict):
                sl = ev["diff"].at(e["ts"]).is_sunlit(EPH) or e["Name"] in ["Erscheint", "Verschwindet"]
                nicht_horizont = not e["Name"] in ["Aufgang", "Untergang"]
                m = e["mag"]
                hell = False
                try:
                    m = float(m)
                    m_str = runde(m, 1, 4)
                    hell = m<sat_mag
                except:
                    m_str = f"{SPACE}?{SPACE*2}"
                if sl:
                    parts.append(dict())
                else:
                    continue
                parts[-1]["dt"] = e["dt"]
                parts[-1]["text"] = f"""<b>{e['Name']}{SPACE*(13-len(e['Name']))}{ptime(e['dt'])}{"</b>" if not hell else ""}{SPACE*2}{m_str}mag{"</b>" if hell else ""}
                                    {SPACE}az:{runde(e['altaz'][1].degrees, 0, 4)}º {rich(e['altaz'][1].degrees)}{SPACE*2}"""
                if nicht_horizont:
                    parts[-1]["text"] += f"{'<b>' if e['Name'] == 'Kulmination' else ''}h: {runde(e['altaz'][0].degrees, 0, 3)}º{'</b>' if e['Name'] == 'Kulmination' else ''}"
                if e["Name"] == "Kulmination":
                    parts[-1]["text"] += f"""<br>{SPACE*2}Distanz: {round(e['Distanz'].km, 1)}km{SPACE*2}Geschw.: {round(e['Geschwindigkeit'].km_per_s, 1)}km/s{SPACE*2}
                    Sonne: {round(ortE.at(ev["Kulmination"]["ts"]).observe(EPH["SUN"]).apparent().altaz()[0].degrees, 1)}º"""
                parts[-1]["text"] += "<br>"

        parts.sort(key = lambda x: x["dt"])
        
        rows.append(dict())
        rows[i]["dt"] = ev['Kulmination']["dt"]
        rows[i]["html"] = html_row(
            ptime(rows[i]["dt"]),
            sat_darstell(ev["satellite"]),
            f'''<img src="{PATH}/tmp/sat{ev['index']}.png" height="90" align="right">{"".join(part["text"] for part in parts)}'''.replace("\n", ""))
    return rows

def draw_sat_überflug(ax:plt.axis, e, lon:float, lat:float,
        tick:timedelta=timedelta(seconds=15)):

    ort, ortE = position(lon, lat)
    minute = timedelta(seconds = 60)
    fm = e["Aufgang"]["dt"]-timedelta(seconds = e["Aufgang"]["dt"].second)+minute

    tag = "#ffe0a5" ## Tag
    colors = ["#665c49", "#282620", "#100f0e", "#060606"]
    sh = ortE.at(e["Kulmination"]["ts"]).observe(EPH["SUN"]).apparent().altaz()[0].degrees
    for col, deg in zip(colors, [0, -6, -12, -18]):
        if sh < deg:
            color = col
    
    if sh < -18 and ortE.at(e["Kulmination"]["ts"]).observe(EPH["MOON"]).apparent().altaz()[0].degrees > 0:
        color = "#051f1f"
        
    ax.set_facecolor(color)

    for zeit in [e["Aufgang"]["dt"]+i*tick for i in range(int((e["Untergang"]["dt"] - e["Aufgang"]["dt"])/tick))]:
        c = satcol(ts.from_datetime(zeit), e["satellite"], lon, lat)
        ax.plot([e["diff"].at(ts.from_datetime(z)).altaz()[1].degrees*pi/180 for z in [zeit, zeit+tick]],
                  [e["diff"].at(ts.from_datetime(z)).altaz()[0].degrees for z in [zeit, zeit+tick]],
                  color = c, lw = 5)

def draw_all_sats(events, lon:float, lat:float):
    plt.rcParams['figure.figsize'] = [4.0, 4.0]
    if not os.path.exists(f"{PATH}/tmp"):
        print("Erstelle Hilfsordner für die Bilder")
        os.mkdir(f"{PATH}/tmp")
    for i, e in tqdm(enumerate(events), desc = "erzeuge Grafiken", total = len(events)):
        fig = plt.figure()
        ax = fig.add_subplot(polar = True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(90, 0)
        ax.set_yticks([0, 45])
        ax.set_yticklabels(["", ""])
        ax.set_xticks([0, np.pi/2, np.pi, np.pi*1.5])
        ax.set_xticklabels(["N", "O", "S", "W"])
        draw_sat_überflug(ax, e, lon, lat, tick = timedelta(seconds = 20))
        ax.grid()
        plt.tight_layout()
        plt.savefig(f"{PATH}/tmp/sat{e['index']}.png")
        plt.cla()

def calsky(dt0:datetime, dt1:datetime, lat:float, lon:float, elev:float, name:str="table",
        sat = None, sat_mag = 4 , mond:bool = True, planeten:bool = True, sonne:bool = True):

    ts0, ts1 = ts.from_datetime(dt0), ts.from_datetime(dt1)
    tab = list()
    if sat:
        sats = satellite_events(sat, ts0, ts1, 20, lon = lon, lat = lat, sun_deg = -6, max_mag = sat_mag)
        draw_all_sats(sats, lon, lat)
        tab += sat_events_to_html(sats, lon, lat, sat_mag = sat_mag)
    if mond:
        tab += mond_events(ts0, ts1, lon, lat, elev)
    if planeten:
        tab += planeten_events(ts0, ts1, lon, lat, elev)
    if sonne:
        tab += sonne_events(ts0, ts1, lon, lat)

    tab.sort(key = lambda x: x["dt"])

    table_head = '<table style="font-family: monospace; width: 80%;margin-left:auto;margin-right:auto"><tr><th witdh="15%">Zeit</th><th width = "15%">Objekt</th><th>Beschreibung</th></tr>'
    def table_caption(zeit:datetime)->str:
        return f'<caption><h3>{zeit.strftime("%A, %Y-%m-%d")}</h3></caption>'

    h = "<!DOCTYPE html>"
    h += "<head>"
    h += "<title>SkyCalc.py</title>"
    h += "<style>"
    h += "table, th, td{border: 1px solid black;border-collapse: collapse;}"
    h += "td{padding: 5px; text-align: left;}"
    h += "</style>"
    h += "</head>"
    h += '<body>'
    h += f'<h1>Astronomische Ereignisse für {round(lat, 2)}ºN, {round(lon, 2)}ºE</h1>'
    h += f"<h3>Daten von {dt0.strftime('%Y-%m-%d um %Hh%Mm%Ss')}</h3>"
    h += f"{table_head}"
    h += f"{table_caption(dt0)}"
    for i, t in enumerate(tab):
        h += t["html"]
        if ((tab[min((i+1), len(tab)-1)]["dt"]).astimezone(TZ).day
                != (t["dt"]).astimezone(TZ).day):
            h +="</table>"
            h += f"{table_head}"
            h += f'{table_caption(tab[(i+1)%len(tab)]["dt"])}'
    h +="</table></body>"
    with open(f"{PATH}/table.html", "w") as f:
        f.write(h)
    print("Fertig")


def main():
    sat = (VISUAL if "-sat" in sys.argv else False)
    dur = 24
    sat_mag = 5
    op = False
    for i, arg in enumerate(sys.argv):
        if arg == "-dur":
            dur = int(sys.argv[i+1])
        if arg == "-sat-mag":
            sat_mag = int(sys.argv[i+1])
            sat = VISUAL
        if arg == "-open":
            op = True

    plt.rcParams['figure.max_open_warning'] = 200 # Warnung unterdrücken
    calsky(datetime.now(TZ), datetime.now(TZ)+timedelta(hours = dur),51.86,7.49,60,"table", sat, sat_mag = sat_mag)
    if op:
        os.system(f"firefox {PATH}/table.html")

if __name__ == "__main__":
    main()
