import { useEffect, useRef, useState } from "react";
import { CircleMarker, MapContainer, TileLayer, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";
const REQUIRED_CSV_COLUMNS_WITH_LONLAT = [
  "Sampling date",
  "Sampling grid",
  "Moisture",
  "Total nitrogen (%)",
  "Total carbon (%)",
  "pH",
  "Organic matter (%)",
  "Aluminum (mg/Kg)",
  "Calcium (mg/Kg)",
  "Copper (mg/Kg)",
  "Iron (mg/Kg)",
  "Potassium (mg/Kg)",
  "Magnesium (mg/Kg)",
  "Manganese (mg/Kg)",
  "Molybdenum (mg/Kg)",
  "Sodium (mg/Kg)",
  "Phosphorus (mg/Kg)",
  "Sulfur (mg/Kg)",
  "Zinc (mg/Kg)",
];

const EXAMPLE_SOIL_FILES = [
  { label: "Sample 1", href: "/examples/Example-1-soil.csv" },
  { label: "Sample 2", href: "/examples/example-2-soil.csv" },
  { label: "Sample 3", href: "/examples/example-3-soil.csv" },
  { label: "Sample 4", href: "/examples/example-4-soil.csv" },
  { label: "Sample 5", href: "/examples/example-5-soil.csv" },
  // { label: "Sample 6", href: "/examples/example-6-soil.csv" },
  // { label: "Sample 7", href: "/examples/example-7-soil.csv" },
];

{/*part where it looks for clicks on the map, and reacts by getting the longitude and latitude selected*/}
function LocationPicker({ setLatInput, setLonInput }) {
  useMapEvents({
    click(e) {
      setLatInput(e.latlng.lat.toFixed(6));
      setLonInput(e.latlng.lng.toFixed(6));
    },
  });
  return null;
}

{/*Setting main variables and default states*/}
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
  const [showCsvHelp, setShowCsvHelp] = useState(false);
  const [requestProgress, setRequestProgress] = useState(0);
  const [requestStage, setRequestStage] = useState("Idle");
  const [successfulCalls, setSuccessfulCalls] = useState(0);
  const fileInputRef = useRef(null);

  const parsedLat = Number(latInput);
  const parsedLon = Number(lonInput);
  {/*how deciding if long and lat are vallid inputs*/}
  const validLat = latInput !== "" && !Number.isNaN(parsedLat) && parsedLat >= -90 && parsedLat <= 90;
  const validLon = lonInput !== "" && !Number.isNaN(parsedLon) && parsedLon >= -180 && parsedLon <= 180;
  const hasValidCoordinates = validLat && validLon;

  {/*Clearing the coordinates button function*/}
  const clearCoordinates = () => {
    setLatInput("");
    setLonInput("");
  };

  const clearCsvFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  {/*toggling between light mode and dark mode*/}
  useEffect(() => {
    document.body.classList.toggle("dark-mode", theme === "dark");
    return () => document.body.classList.remove("dark-mode");
  }, [theme]);

  {/*handling the connection to the backend*/}
  const handleSubmit = async () => {
    {/*providing user feedback*/}
    setResult("");
    setLoading(true);
    setRequestProgress(10);
    setRequestStage("Preparing request...");

    const formData = new FormData();

    if (file) formData.append("file", file);

    {/*User feedback. (Needto have at least soil or long/lat data.*/}
    if (hasValidCoordinates) {
      formData.append("lat", parsedLat);
      formData.append("lon", parsedLon);
      setRequestProgress(25);
      setRequestStage("Coordinates queued. Waiting for GIS...");
    } else if (!file) {
      setResult("Error: Add a CSV and/or enter valid latitude and longitude.");
      setLoading(false);
      setRequestProgress(0);
      setRequestStage("Idle");
      return;
    }

    {/*inputting the data to push to the backend*/}
    formData.append("model_type", modelType);
    formData.append("longlat_mode", longlatMode);

    try {
      {/*Connection to backend*/}
      setRequestProgress(40);
      setRequestStage("Sending API request...");
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      {/*reads server's json into data*/}
      setRequestProgress(70);
      setRequestStage("Processing response...");
      const data = await response.json();

      {/*checking if the http succeded, and if the backend operation succeeded*/}
      if (response.ok && data.success) {
        if (data.gis_loaded) {
          const ms = data.gis_fetch_ms ?? "?";
          setRequestProgress(90);
          setRequestStage(`GIS loaded (${ms} ms). Finalizing model output...`);
        }

        {/*building object with prediction result and NLCD percentages! so that it can be displayed*/}
        setResult(
          JSON.stringify(
            {
              result: data.result,
              nlcd_percentages: data.nlcd_percentages,
            },
            null,
            2
          )
        );
        setRequestProgress(100);
        setRequestStage("Complete");
        setSuccessfulCalls((prev) => prev + 1);
      } else {
        {/*Error checks*/}
        setResult(`Error: ${data.detail || data.error || "Unknown error"}`);
        setRequestProgress(0);
        setRequestStage("Request failed");
      }
    } catch {
      setResult("Server connection failed.");
      setRequestProgress(0);
      setRequestStage("Server connection failed");
    }
    {/*turns off loading state (button and UI resets to normal)*/}
    setLoading(false);
  };

  {/*UI interface!!!! Yay*/}
  return (
    <div className={`app-shell ${theme === "dark" ? "theme-dark" : "theme-light"}`}>
      <div className="hero-panel">
        <div className="hero-top-row">
          <p className="kicker">Field-Listeria Intelligence</p>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "Light Mode" : "Dark Mode"}
          </button>
        </div>
        {/* Here is where the Tool name and tool description is*/}
        <h1>Listeria Risk Tool</h1>
        <p className="hero-copy">
          The goal of this project is to help you identify the risk level of your field.

          Upload a field dataset and/or set a location to score pathogen risk and prediction outputs. Then choose a prediction model or a risk model.

          If you want to predict your likelihood of Listeria contamination, use the prediction model.
          If you want to predict the risk of Listeria in your field, then select the risk model. 

          For the best results, include both a coordinate selection and soil test results. 
          
          If you do not have soil results yet, and you want help deciding if getting a soil test would help you identify Listeria risk, then select the risk model, and just run it with longitude and latitude.
        </p>
      </div>

      {/* Here is where the data inputs are*/}
      <div className="grid">
        <section className="card">
          <h2>Data Input</h2>
          {/*here is the csv upload section*/}
          <label className="label">CSV Upload (optional)</label>
          <div className="csv-upload-row">
            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0])}
            />
            <button
              type="button"
              className={`chip csv-inline-help-icon ${showCsvHelp ? "active" : ""}`}
              onClick={() => setShowCsvHelp((prev) => !prev)}
              aria-label={showCsvHelp ? "Hide CSV help" : "Show CSV help"}
              title={showCsvHelp ? "Hide CSV help" : "Show CSV help"}
            >
              ?
            </button>
          </div>
          <div className="csv-tools-row">
            <button
              type="button"
              className="chip clear"
              onClick={clearCsvFile}
            >
              Clear CSV
            </button>
            {EXAMPLE_SOIL_FILES.map((sample) => (
              <a key={sample.href} className="chip sample-link" href={sample.href} download>
                {sample.label}
              </a>
            ))}
          </div>
          {showCsvHelp && (
            <div className="csv-help-box">
              <p className="csv-help-title">Required CSV columns</p>
              <p className="csv-help-note">
                When inputting a CSV, make sure all of the files have the following information.
              </p>
              <pre className="csv-help-list">{REQUIRED_CSV_COLUMNS_WITH_LONLAT.join("\n")}</pre>
            </div>
          )}

          {/*here is the location entry option. There are three main buttons: 
          (1) a map where you can click the location you want
          (2) a manual entry option where you can 
          (3) a clear coordinate section (to clear the part where you entered a location on a map*/}
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

          {/*here is the part where it fills in an example of the manual*/}
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
              {/*here is the map part where it centers on the US*/}
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
              <p className="hint">Click anywhere on the map to set coordinates. Click and drag to move the map, and zoom in or out using the buttons as needed.</p>
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

          <label className="label">Model Mode (with or without soil/coordinates)</label>
          <select className="select-input" value={longlatMode} onChange={(e) => setLonglatMode(e.target.value)}>
            <option value="with_longlat">Latitude and Longitude Included</option>
            <option value="with_soil">Soil Information Only</option>
            <option value="soil_longlat">Both Soil and Latitude Longitude Information</option>
          </select>

          {/*some user feedback*/}
          <button className="run-btn" onClick={handleSubmit} disabled={loading}>
            {loading ? "Running..." : "Run Model"}
          </button>
          <div className="progress-meta">
            <span>Status: {requestStage}</span>
            <span>Successful API Calls: {successfulCalls}</span>
          </div>
          <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={requestProgress}>
            <div className="progress-fill" style={{ width: `${requestProgress}%` }} />
          </div>

          {/*results section*/}
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
