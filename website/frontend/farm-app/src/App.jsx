import { useEffect, useRef, useState } from "react";
import { CircleMarker, MapContainer, TileLayer, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";
const REQUIRED_CSV_COLUMNS_WITH_LONLAT = [
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
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [locationMode, setLocationMode] = useState("map");
  const [latInput, setLatInput] = useState("");
  const [lonInput, setLonInput] = useState("");
  const [forecastDateInput, setForecastDateInput] = useState("");
  const [modelType, setModelType] = useState("gbm");
  const [longlatMode, setLonglatMode] = useState("longlat_only");
  const [irrigationMode, setIrrigationMode] = useState("none");
  const [wildlifeMode, setWildlifeMode] = useState("none");
  const [manureMode, setManureMode] = useState("none");
  const [bufferZoneMode, setBufferZoneMode] = useState("none");
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

  const parseForecastDate = (raw) => {
    const match = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(raw.trim());
    if (!match) return null;
    const month = Number(match[1]);
    const day = Number(match[2]);
    const year = Number(match[3]);
    const date = new Date(Date.UTC(year, month - 1, day));
    const validDate =
      date.getUTCFullYear() === year &&
      date.getUTCMonth() === month - 1 &&
      date.getUTCDate() === day;
    return validDate ? date : null;
  };

  const isForecastDateAllowed = (dateUtc) => {
    const now = new Date();
    const todayUtc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const maxUtc = new Date(todayUtc);
    maxUtc.setUTCDate(maxUtc.getUTCDate() + 14);
    return dateUtc <= maxUtc;
  };

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
    const requiresCoordinates = longlatMode !== "soil_only";
    if (requiresCoordinates) {
      if (!hasValidCoordinates) {
        setResult("Error: This model mode requires valid latitude and longitude.");
        setLoading(false);
        setRequestProgress(0);
        setRequestStage("Missing coordinates");
        return;
      }

      const parsedForecastDate = parseForecastDate(forecastDateInput);
      if (!parsedForecastDate) {
        setResult("Error: Enter forecast date as MM/DD/YYYY.");
        setLoading(false);
        setRequestProgress(0);
        setRequestStage("Invalid date format");
        return;
      }
      if (!isForecastDateAllowed(parsedForecastDate)) {
        setResult("Error: Date is too far ahead. Choose historical/today or up to 14 days into the future (UTC).");
        setLoading(false);
        setRequestProgress(0);
        setRequestStage("Date out of bounds");
        return;
      }
      formData.append("lat", parsedLat);
      formData.append("lon", parsedLon);
      formData.append("forecast_date", forecastDateInput.trim());
      setRequestProgress(25);
      setRequestStage("Coordinates queued. Waiting for GIS...");
    } else if (!file) {
      setResult("Error: Soil-only mode requires a CSV upload.");
      setLoading(false);
      setRequestProgress(0);
      setRequestStage("Missing CSV");
      return;
    }

    {/*inputting the data to push to the backend*/}
    formData.append("model_type", modelType);
    formData.append("longlat_mode", longlatMode);
    formData.append("irrigation_mode", irrigationMode);
    formData.append("wildlife_mode", wildlifeMode);
    formData.append("manure_mode", manureMode);
    formData.append("buffer_zone_mode", bufferZoneMode);

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
              model_type: data.model_type,
              nlcd_percentages: data.nlcd_percentages,
              weather_data: data.weather_data,
              forecast_date_utc: data.forecast_date_utc,
              irrigation_mode: data.irrigation_mode,
              wildlife_mode: data.wildlife_mode,
              manure_mode: data.manure_mode,
              buffer_zone_mode: data.buffer_zone_mode,
              probability_presence_base: data.probability_presence_base,
              probability_presence_adjusted: data.probability_presence_adjusted,
              to_return_risk_class: data.to_return_risk_class,
              displayed_result: data.displayed_result,
              add_message: data.add_message,
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
          <div className="hero-controls">
            <button
              type="button"
              className="theme-toggle"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? "Light Mode" : "Dark Mode"}
            </button>
            <button
              type="button"
              className="help-toggle"
              onClick={() => setShowHelpModal(true)}
              aria-label="Open help"
              title="Open help"
            >
              ?
            </button>
          </div>
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

      {/* this section is the help button at the top right corner (main how to help) */}
      {showHelpModal && (
        <div className="help-modal-overlay" role="dialog" aria-modal="true" aria-label="Help instructions">
          <div className="help-modal">
            <button
              type="button"
              className="help-exit-btn"
              onClick={() => setShowHelpModal(false)}
              aria-label="Close help"
              title="Close help"
            >
              &times;
            </button>
            <h3>Uploading and Entering Data and Running the Predictive Model / Risk Score</h3>
            <p>To run the predictive model please do 1 of the following:</p>
            <ul>
              <li>
                To run a soil-only model:
                <ol>
                  <li>Upload a CSV in the "Soil CSV Upload (optional)" Section</li>
                  <li>
                    In the "Model Mode (with or without soil/coordinates)" Section, click the dropdown and select
                    "Soil Information Only".
                  </li>
                  <li>
                    In "Model Type", select any of the options ("Gradient Boosted Model (Recommended and Best)",
                    "Neural Network", or "SVM (Support Vector Machine)")
                  </li>
                </ol>
              </li>
              <li>
                To run a model from longitude and latitude data only (weather and elevation data retrieved from an API automatically)
                <ol>
                  <li>
                    Enter a date of interest (any time after 2010, and up to 14 days in the future) in the Month/Day/Year
                    format (i.e. 02/14/2026). Please note the current day's data and future data will be retrieved through
                    a forcasting model, and could be innacurate and affect model results.
                  </li>
                  <li>
                    Select a point on the map or manually enter a coordinate using the "Choosing Coordinates" section.
                    <ul>
                      <li>
                        To enter a point manually, press the "Manual Entry" button and input your longitude and latitude.
                      </li>
                      <li>
                        To select a point on the map, press the "Map Pick" button and drag/zoom in as needed to select your
                        location on a map. The selected coordinates will be displayed below the map.
                      </li>
                    </ul>
                  </li>
                  <li>
                    In the "Model Mode (with or without soil/coordinates)" Section, click the dropdown and select
                    "Latitude and Longitude Information Only".
                  </li>
                  <li>
                    In "Model Type", select any of the options ("Gradient Boosted Model (Recommended and Best)", "Neural Network",
                    or "SVM (Support Vector Machine)")
                  </li>
                </ol>
              </li>
              <li>
                To run a model with both soil data and longitude/latitude data, please follow the instructions of the previous
                two sections (add soil and longitude/latitude data)
              </li>
            </ul>
            <p>
              For more information on the soil data requirements, please select the help button, and/or dowload the CSVs provided.
            </p>
            <p>
              For information on how to go from an excel file to a CSV file, please go to this website:{" "}
              <a
                href="https://support.microsoft.com/en-us/office/save-a-workbook-to-text-format-txt-or-csv-3e9a9d6c-70da-4255-aa28-fcacf1f081e6"
                target="_blank"
                rel="noreferrer"
              >
                https://support.microsoft.com/en-us/office/save-a-workbook-to-text-format-txt-or-csv-3e9a9d6c-70da-4255-aa28-fcacf1f081e6
              </a>
            </p>
            <p>and in the specified area, please select the "CSV (comma delimited)" option.</p>
          </div>
        </div>
      )}

      {/* Here is where the data inputs are*/}
      <div className="grid">
        <section className="card">
          <h2>Data Inputs</h2>
          {/*here is the csv upload section*/}
          <label className="label">Soil CSV Upload (optional)</label>
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

          
          {/* Here is where the date information is uploaded */}
          <label className="label">Date (MM/DD/YYYY, historical or max +14 days)</label>
          <input
            className="text-input"
            type="text"
            placeholder="MM/DD/YYYY"
            inputMode="numeric"
            pattern="\d{2}/\d{2}/\d{4}"
            value={forecastDateInput}
            onChange={(e) => setForecastDateInput(e.target.value)}
          />
          <p className="hint">Required for coordinate-based runs. Supports historical dates (after 2010) and up to 14 days ahead (UTC).</p>

          {/*here is the location entry option. There are three main buttons: 
          (1) a map where you can click the location you want
          (2) a manual entry option where you can 
          (3) a clear coordinate section (to clear the part where you entered a location on a map*/}
          <label className="label">Choosing Coordinates</label>
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

        {/* The little helper that automatically updates with long-lat selected. a cute little touch for user feedback. */}
        <div className="coord-readout">
            <span>Latitude: {latInput || "--"}</span>
            <span>Longitude: {lonInput || "--"}</span>
          </div>

        </section>

        <section className="card">
          <h2>Model Control</h2>
          <label className="label">Model Type (which model variant to use)</label>
          <select className="select-input" value={modelType} onChange={(e) => setModelType(e.target.value)}>
            <option value="gbm">Gradient Boosted Model (Recommended and Best Performance)</option>
            <option value="neural_net">Neural Network (Takes a long time, but comparable model performance to the Gradient Boosted Model)</option>
            <option value="svm">Support Vector Machine Model (Lowest accuracy performance, but still comparable to the Gradient Boosted Model)</option>
          </select>

          <label className="label">Model Mode (type of data you want to run the model on)</label>
          <select className="select-input" value={longlatMode} onChange={(e) => setLonglatMode(e.target.value)}>
            <option value="longlat_only">Latitude and Longitude Information Only</option>
            <option value="soil_only">Soil Information Only</option>
            <option value="soil_longlat">Both Soil and Latitude Longitude Information</option>
          </select>

          <label className="label">Irrigation (optional)</label>
          <select className="select-input" value={irrigationMode} onChange={(e) => setIrrigationMode(e.target.value)}>
            <option value="none">No selection</option>
            <option value="24_rain_window">24 hours since last rain/irrigation</option>
            <option value="48_rain_window">48 hours since last rain/irrigation</option>
            <option value="72_rain_window">72 hours since last rain/irrigation</option>
            <option value="144_rain_window">144+ hours since last rain/irrigation</option>
          </select>

          <label className="label">Wildlife Traffic (optional)</label>
          <select className="select-input" value={wildlifeMode} onChange={(e) => setWildlifeMode(e.target.value)}>
            <option value="none">No selection</option>
            <option value="no_risk_wildlife">No wildlife seen in field</option>
            <option value="low_risk_wildlife">Wildlife seen 8-30 days ago</option>
            <option value="moderate_risk_wildlife">Wildlife seen 4-7 days ago</option>
            <option value="high_risk_wildlife">Wildlife seen within last 3 days</option>
          </select>

          <label className="label">Manure (optional)</label>
          <select className="select-input" value={manureMode} onChange={(e) => setManureMode(e.target.value)}>
            <option value="none">No selection</option>
            <option value="no_manure">Manure never spread on field</option>
            <option value="manure_over_365_days">Manure spread over 365 days before harvest</option>
            <option value="manure_within_365_days">Manure spread within 365 days of harvest</option>
          </select>

          <label className="label">Buffer Zone (optional)</label>
          <select className="select-input" value={bufferZoneMode} onChange={(e) => setBufferZoneMode(e.target.value)}>
            <option value="none">No selection</option>
            <option value="no_buffer_zone">No buffer zone</option>
            <option value="buffer_zone">Buffer zone present</option>
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
