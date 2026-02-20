import { useEffect, useState } from "react";
import { CircleMarker, MapContainer, TileLayer, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

function LocationPicker({ setLatInput, setLonInput }) {
  useMapEvents({
    click(e) {
      setLatInput(e.latlng.lat.toFixed(6));
      setLonInput(e.latlng.lng.toFixed(6));
    },
  });
  return null;
}


function App() {
  const [theme, setTheme] = useState("light");
  const [locationMode, setLocationMode] = useState("map");
  const [latInput, setLatInput] = useState("");
  const [lonInput, setLonInput] = useState("");
  const [modelType, setModelType] = useState("prediction");
  const [longlatMode, setLonglatMode] = useState("with_longlat");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const parsedLat = Number(latInput);
  const parsedLon = Number(lonInput);
  const validLat = latInput !== "" && !Number.isNaN(parsedLat) && parsedLat >= -90 && parsedLat <= 90;
  const validLon = lonInput !== "" && !Number.isNaN(parsedLon) && parsedLon >= -180 && parsedLon <= 180;
  const hasValidCoordinates = validLat && validLon;

  const clearCoordinates = () => {
    setLatInput("");
    setLonInput("");
  };

  useEffect(() => {
    document.body.classList.toggle("dark-mode", theme === "dark");
    return () => document.body.classList.remove("dark-mode");
  }, [theme]);

  const handleSubmit = async () => {
    setResult("");
    setLoading(true);

    const formData = new FormData();

    if (file) formData.append("file", file);

    if (hasValidCoordinates) {
      formData.append("lat", parsedLat);
      formData.append("lon", parsedLon);
    } else if (!file) {
      setResult("Error: Add a CSV or enter valid latitude and longitude.");
      setLoading(false);
      return;
    }

    formData.append("model_type", modelType);
    formData.append("longlat_mode", longlatMode);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setResult(JSON.stringify(data.result));
      } else {
        setResult(`Error: ${data.detail || data.error || "Unknown error"}`);
      }
    } catch {
      setResult("Server connection failed.");
    }

    setLoading(false);
  };

  return (
    <div className={`app-shell ${theme === "dark" ? "theme-dark" : "theme-light"}`}>
      <div className="hero-panel">
        <div className="hero-top-row">
          <p className="kicker">Field Intelligence</p>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "Light Mode" : "Dark Mode"}
          </button>
        </div>
        <h1>Farm Decision Tool</h1>
        <p className="hero-copy">
          Upload a field dataset or set a location to score pathogen risk and prediction outputs.
        </p>
      </div>

      <div className="grid">
        <section className="card">
          <h2>Data Input</h2>
          <label className="label">CSV Upload (optional)</label>
          <input className="file-input" type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />

          <label className="label">Location Entry</label>
          <div className="toggle-row">
            <button
              type="button"
              className={`chip ${locationMode === "map" ? "active" : ""}`}
              onClick={() => setLocationMode("map")}
            >
              Map Pick
            </button>
            <button
              type="button"
              className={`chip ${locationMode === "manual" ? "active" : ""}`}
              onClick={() => setLocationMode("manual")}
            >
              Manual Entry
            </button>
            <button type="button" className="chip clear" onClick={clearCoordinates}>
              Clear Coordinates
            </button>
          </div>

          {locationMode === "manual" ? (
            <div className="manual-grid">
              <div>
                <label className="label">Latitude</label>
                <input
                  className="text-input"
                  type="number"
                  step="0.000001"
                  min="-90"
                  max="90"
                  placeholder="ex: 42.453449"
                  value={latInput}
                  onChange={(e) => setLatInput(e.target.value)}
                />
              </div>
              <div>
                <label className="label">Longitude</label>
                <input
                  className="text-input"
                  type="number"
                  step="0.000001"
                  min="-180"
                  max="180"
                  placeholder="ex: -76.473503"
                  value={lonInput}
                  onChange={(e) => setLonInput(e.target.value)}
                />
              </div>
            </div>
          ) : (
            <div className="map-wrap">
              <MapContainer center={[39, -95]} zoom={4} className="map">
                <TileLayer
                  attribution="&copy; OpenStreetMap"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <LocationPicker setLatInput={setLatInput} setLonInput={setLonInput} />
                {hasValidCoordinates && (
                  <CircleMarker center={[parsedLat, parsedLon]} radius={8} pathOptions={{ color: "#f56300" }} />
                )}
              </MapContainer>
              <p className="hint">Click anywhere on the map to set coordinates.</p>
            </div>
          )}

          <div className="coord-readout">
            <span>Latitude: {latInput || "--"}</span>
            <span>Longitude: {lonInput || "--"}</span>
          </div>
        </section>

        <section className="card">
          <h2>Model Control</h2>
          <label className="label">Model Type</label>
          <select className="select-input" value={modelType} onChange={(e) => setModelType(e.target.value)}>
            <option value="prediction">Prediction Model</option>
            <option value="risk">Risk Model</option>
          </select>

          <label className="label">Long/Lat Mode</label>
          <select className="select-input" value={longlatMode} onChange={(e) => setLonglatMode(e.target.value)}>
            <option value="with_longlat">With Long/Lat</option>
            <option value="without_longlat">Without Long/Lat (CSV only)</option>
          </select>

          <button className="run-btn" onClick={handleSubmit} disabled={loading}>
            {loading ? "Running..." : "Run Model"}
          </button>

          <div className="result-box">
            <h3>Result</h3>
            <p>{result || "No result yet."}</p>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
