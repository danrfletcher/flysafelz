"""Read-only map data routes for the UI: GET /map/airspace, GET /map/notams and GET /map/wind.

    config.py     the project root, the UI origins CORS allows, and the openAIP key (read through capture.find_key)
    cache.py      the disk cache under data/map/, its freshness rule, and the one answer a route gives (200 fresh, 200 stale, 503)
    airspace.py   every GB airspace from openAIP as GeoJSON features, and the limit text and feet
    notams.py     the live PIB's NOTAMs active from now to 24 h ahead as GeoJSON features
    wind.py       Open-Meteo's current 10 m wind on a 1 degree grid over the UK as leaflet-velocity's U and V records
    app.py        the Starlette app that langgraph.json mounts as `http.app`

This package is the one place under src/flysafelz that imports a web framework and the one (with reader/) where the word `cache` is
allowed (fence F6). It writes only under data/map/ (F8) and never logs a NOTAM's text or the openAIP key (Q10).
"""
