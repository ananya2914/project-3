#!/bin/sh
# Rebuild report.pdf from report_src/report.html (includes ../perfetto.png if present)
cd "$(dirname "$0")"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="../report.pdf" "file://$PWD/report.html" 2>/dev/null
ls -la ../report.pdf
