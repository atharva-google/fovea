# Fovea: the web app

A page in your browser, served by a small program on this computer. Nothing is sent anywhere except
the video copy and text that Gemini needs.

```bash
python frontend/server.py
```

The page opens at http://127.0.0.1:8765/. Add a video, paste a Gemini key under Settings, and Fovea
watches the video once. Then:

- pick a format to see what it keeps in the picture, and drag the window to move it for a shot
- untick shots to leave them out of renders and edits
- **Render** makes every format; the **Edit** tab makes a shorter version you can adjust before rendering

The program stops by itself after 10 idle minutes (never during a job). Start it again to continue;
the page reconnects by itself.

```
frontend/
├── server.py     the local server: serves the page, runs the steps (standard library only)
├── index.html    the page
├── style.css     the dark theme
├── app.js        what the page does
└── sw.js         lets the page show its "asleep" message when the server is not running
```

What the page adds to a project is stored next to it, in `fovea_data/<video name>/frontend.json`.
The Gemini key is stored in `frontend/.local/` (not in git).
