Place your Google Earth Engine service-account JSON key here with this exact name:

gee-service-account-key.json

Then run, from repo root:

powershell -ExecutionPolicy Bypass -File .\setup_competition.ps1

Or on macOS/Linux:

chmod +x ./setup_competition.sh
./setup_competition.sh

Do not commit JSON keys to git.
