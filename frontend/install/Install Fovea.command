#!/bin/bash
# Fovea setup for Mac and Linux. Run it once (double-click on a Mac); run it again to update Fovea.
# It needs nothing installed: it brings its own Python, libraries and ffmpeg, all inside one folder.
# To remove Fovea, delete that folder and the Fovea icon.
set -euo pipefail

ROOT="${FOVEA_INSTALL_DIR:-$HOME/Fovea}"
BRANCH="${FOVEA_BRANCH:-feat/frontend}"
REPO="https://github.com/atharva-google/fovea"

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }
trap 'printf "\nSetup did not finish. Check your internet connection and run this file again.\n"' ERR

mkdir -p "$ROOT"
say "1/5  Python tools"
export UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1
[ -x "$ROOT/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV="$ROOT/bin/uv"

say "2/5  Fovea itself"
TMP="$(mktemp -d)"
curl -LsSf "$REPO/archive/refs/heads/$BRANCH.tar.gz" | tar -xz -C "$TMP"
rm -rf "$ROOT/app"
mv "$TMP"/* "$ROOT/app"
rm -rf "$TMP"
# which version this is, so the Update button knows when there is a newer one
(curl -fsSL "https://api.github.com/repos/atharva-google/fovea/commits/$BRANCH" | grep -m1 '"sha"' | cut -d'"' -f4 \
  > "$ROOT/version") 2>/dev/null || rm -f "$ROOT/version"

say "3/5  Python and the libraries (a few minutes the first time)"
[ -d "$ROOT/venv" ] || "$UV" venv --python 3.12 "$ROOT/venv"
"$UV" pip install --python "$ROOT/venv/bin/python" -r "$ROOT/app/requirements.txt" static-ffmpeg

say "4/5  The speech model and ffmpeg"
"$ROOT/venv/bin/python" "$ROOT/app/frontend/launch.py" --prepare

say "5/5  The Fovea icon"
if [ -n "${FOVEA_NO_ICON:-}" ]; then
  echo "skipped"
elif [ "$(uname)" = "Darwin" ]; then
  APP="$HOME/Applications/Fovea.app"
  PY="$ROOT/venv/bin/python"; LAUNCH="$ROOT/app/frontend/launch.py"
  mkdir -p "$HOME/Applications"
  rm -rf "$APP"
  # A small real Mac app (an AppleScript applet) that starts Fovea on its own and quits at once. Because
  # it quits, macOS runs it again on every click, and launch.py then opens the page if Fovea is awake.
  # (An app that is only a shell script is not tracked properly by macOS: a second click does nothing,
  # or says "The application Fovea is not open anymore".)
  if ! osacompile -o "$APP" -e "do shell script \"nohup \" & quoted form of \"$PY\" & \" \" & quoted form of \"$LAUNCH\" & \" >/dev/null 2>&1 &\"" >/dev/null 2>&1; then
    mkdir -p "$APP/Contents/MacOS"                 # no osacompile: a plain script app that also quits at once
    printf '#!/bin/bash\nnohup "%s" "%s" >/dev/null 2>&1 &\n' "$PY" "$LAUNCH" > "$APP/Contents/MacOS/Fovea"
    chmod +x "$APP/Contents/MacOS/Fovea"
    cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Fovea</string>
  <key>CFBundleIdentifier</key><string>local.fovea.app</string>
  <key>CFBundleExecutable</key><string>Fovea</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict></plist>
PLIST
  fi
  (                                               # the icon picture; Fovea works without it
    SET="$(mktemp -d)/Fovea.iconset"; mkdir -p "$SET" "$APP/Contents/Resources"
    for size in 16 32 128 256; do
      sips -z $size $size "$ROOT/app/frontend/icon.png" --out "$SET/icon_${size}x${size}.png"
      sips -z $((size * 2)) $((size * 2)) "$ROOT/app/frontend/icon.png" --out "$SET/icon_${size}x${size}@2x.png"
    done
    iconutil -c icns "$SET" -o "$APP/Contents/Resources/Fovea.icns"
    plutil -replace CFBundleIconFile -string Fovea "$APP/Contents/Info.plist"
    plutil -remove CFBundleIconName "$APP/Contents/Info.plist" || true
    rm -f "$APP/Contents/Resources/Assets.car"
  ) >/dev/null 2>&1 || true
  plutil -replace LSUIElement -bool true "$APP/Contents/Info.plist" >/dev/null 2>&1 || true   # no Dock icon
  codesign --force --deep -s - "$APP" >/dev/null 2>&1 || true      # the app was changed after it was made
  touch "$APP"
  ln -sfn "$APP" "$HOME/Desktop/Fovea"
  echo "Fovea is in your Applications folder and on your Desktop."
else
  ICON="$HOME/.local/share/applications/fovea.desktop"
  mkdir -p "$(dirname "$ICON")"
  printf '[Desktop Entry]\nType=Application\nName=Fovea\nExec="%s" "%s"\nIcon=%s\nTerminal=false\n' \
    "$ROOT/venv/bin/python" "$ROOT/app/frontend/launch.py" "$ROOT/app/frontend/icon.png" > "$ICON"
  chmod +x "$ICON"
  [ -d "$HOME/Desktop" ] && cp "$ICON" "$HOME/Desktop/fovea.desktop"
  echo "Fovea is in your applications menu."
fi

say "Done. Opening Fovea ..."
if [ -z "${FOVEA_NO_ICON:-}" ]; then
  nohup "$ROOT/venv/bin/python" "$ROOT/app/frontend/launch.py" >/dev/null 2>&1 &
fi
echo "From now on, click the Fovea icon. You can close this window."
