-- ============================================================
-- VarshaMitra – Database Schema
-- PostgreSQL + PostGIS
-- ============================================================

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- ─── TABLES ─────────────────────────────────────────────────

-- Cities
CREATE TABLE cities (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    state       VARCHAR(100),
    country     VARCHAR(50) DEFAULT 'India',
    centroid    GEOGRAPHY(POINT, 4326),
    area_km2    NUMERIC(10,2),
    population  BIGINT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Wards (municipal ward boundaries)
CREATE TABLE wards (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    city_id         INTEGER REFERENCES cities(id),
    city            VARCHAR(100),
    ward_number     VARCHAR(20),
    centroid        GEOGRAPHY(POINT, 4326),
    boundary        GEOGRAPHY(POLYGON, 4326),
    area_km2        NUMERIC(8,4),
    population      INTEGER,
    -- Cached elevation from OpenTopoData SRTM30m (reduces API calls)
    cached_elevation_m  NUMERIC(8,2),
    drainage_density    NUMERIC(6,2),
    elevation_source    VARCHAR(50) DEFAULT 'opentopodata_srtm30m',
    elevation_cached_at TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_wards_city ON wards(city_id);
CREATE INDEX idx_wards_boundary ON wards USING GIST(boundary);

-- Rainfall Stations (IMD)
CREATE TABLE rainfall_stations (
    id              SERIAL PRIMARY KEY,
    station_id      VARCHAR(50) UNIQUE,
    name            VARCHAR(200),
    city            VARCHAR(100),
    location        GEOGRAPHY(POINT, 4326),
    elevation_m     NUMERIC(8,2),
    active          BOOLEAN DEFAULT TRUE
);
CREATE INDEX idx_stations_location ON rainfall_stations USING GIST(location);

-- Rainfall Observations
CREATE TABLE rainfall_observations (
    id              BIGSERIAL PRIMARY KEY,
    station_id      INTEGER REFERENCES rainfall_stations(id),
    observed_at     TIMESTAMP NOT NULL,
    rainfall_mm     NUMERIC(8,2) NOT NULL,
    duration_hrs    INTEGER DEFAULT 24,
    source          VARCHAR(50) DEFAULT 'IMD'
);
CREATE INDEX idx_rainfall_time ON rainfall_observations(observed_at DESC);
CREATE INDEX idx_rainfall_station ON rainfall_observations(station_id);

-- Elevation Data (derived from ISRO CartoDEM)
CREATE TABLE ward_elevation (
    id              SERIAL PRIMARY KEY,
    ward_id         INTEGER REFERENCES wards(id),
    mean_elevation  NUMERIC(8,2),
    min_elevation   NUMERIC(8,2),
    max_elevation   NUMERIC(8,2),
    std_elevation   NUMERIC(8,2),
    low_lying_pct   NUMERIC(5,2),   -- % pixels below 10m
    dem_source      VARCHAR(50) DEFAULT 'opentopodata_srtm30m',  -- was ISRO_CartoDEM_30m
    processed_at    TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_elevation_ward ON ward_elevation(ward_id);

-- Drainage Network (OSM)
CREATE TABLE drainage_segments (
    id              SERIAL PRIMARY KEY,
    osm_id          BIGINT,
    ward_id         INTEGER REFERENCES wards(id),
    drain_type      VARCHAR(50),   -- canal, storm_drain, culvert, etc.
    geometry        GEOGRAPHY(LINESTRING, 4326),
    capacity_m3h    NUMERIC(10,2),
    condition       VARCHAR(30),   -- good, fair, poor, blocked
    last_inspected  DATE,
    length_m        NUMERIC(10,2)
);
CREATE INDEX idx_drainage_ward ON drainage_segments(ward_id);
CREATE INDEX idx_drainage_geom ON drainage_segments USING GIST(geometry);

-- Drainage Capacity per Ward (aggregated)
CREATE TABLE ward_drainage (
    id                  SERIAL PRIMARY KEY,
    ward_id             INTEGER REFERENCES wards(id),
    total_drain_length  NUMERIC(10,2),   -- metres
    drainage_density    NUMERIC(6,4),    -- km/km²
    capacity_pct        NUMERIC(5,2),    -- % of ideal capacity
    blocked_count       INTEGER DEFAULT 0,
    assessed_at         TIMESTAMP DEFAULT NOW()
);

-- Flood History (NDMA)
CREATE TABLE flood_events (
    id              SERIAL PRIMARY KEY,
    ward_id         INTEGER REFERENCES wards(id),
    city            VARCHAR(100),
    event_date      DATE NOT NULL,
    duration_hrs    INTEGER,
    severity        VARCHAR(20),       -- major, moderate, minor
    rainfall_mm     NUMERIC(8,2),
    affected_area   GEOGRAPHY(POLYGON, 4326),
    deaths          INTEGER DEFAULT 0,
    displaced       INTEGER DEFAULT 0,
    damage_crores   NUMERIC(10,2),
    source          VARCHAR(100) DEFAULT 'NDMA',
    notes           TEXT
);
CREATE INDEX idx_flood_ward ON flood_events(ward_id);
CREATE INDEX idx_flood_date ON flood_events(event_date DESC);

-- Flood Readiness Scores (computed)
CREATE TABLE flood_scores (
    id                  BIGSERIAL PRIMARY KEY,
    ward_id             INTEGER REFERENCES wards(id),
    score               INTEGER NOT NULL CHECK (score BETWEEN 0 AND 100),
    risk_class          VARCHAR(20) NOT NULL,   -- RED_ALERT, WATCH_ZONE, SAFE_ZONE
    rainfall_mm         NUMERIC(8,2),
    elevation_m         NUMERIC(8,2),
    drainage_pct        NUMERIC(5,2),
    flood_events_5yr    INTEGER,
    rain_risk_component NUMERIC(6,4),
    elev_risk_component NUMERIC(6,4),
    drain_risk_component NUMERIC(6,4),
    hist_risk_component  NUMERIC(6,4),
    composite_risk      NUMERIC(6,4),
    ml_probability      NUMERIC(6,4),
    ml_risk_class       VARCHAR(20),
    computed_at         TIMESTAMP DEFAULT NOW(),
    model_version       VARCHAR(20) DEFAULT 'v2.0'
);
CREATE INDEX idx_scores_ward ON flood_scores(ward_id);
CREATE INDEX idx_scores_time ON flood_scores(computed_at DESC);
CREATE INDEX idx_scores_class ON flood_scores(risk_class);

-- Active Alerts
CREATE TABLE flood_alerts (
    id              BIGSERIAL PRIMARY KEY,
    ward_id         INTEGER REFERENCES wards(id),
    severity        VARCHAR(20) NOT NULL,  -- CRITICAL, WARNING, INFO
    title           VARCHAR(200),
    message         TEXT,
    issued_at       TIMESTAMP DEFAULT NOW(),
    expires_at      TIMESTAMP,
    active          BOOLEAN DEFAULT TRUE,
    issued_by       VARCHAR(100),
    channels        JSONB DEFAULT '["dashboard"]'
);
CREATE INDEX idx_alerts_ward ON flood_alerts(ward_id);
CREATE INDEX idx_alerts_active ON flood_alerts(active);

-- Citizen Drain Reports
CREATE TABLE drain_reports (
    id              BIGSERIAL PRIMARY KEY,
    ticket_id       VARCHAR(30) UNIQUE NOT NULL,
    ward_id         INTEGER REFERENCES wards(id),
    ward_name       VARCHAR(200),
    city            VARCHAR(100),
    location        GEOGRAPHY(POINT, 4326),
    description     TEXT,
    reporter_name   VARCHAR(200),
    reporter_phone  VARCHAR(20),
    severity        VARCHAR(20) DEFAULT 'MEDIUM',
    status          VARCHAR(30) DEFAULT 'SUBMITTED',
    submitted_at    TIMESTAMP DEFAULT NOW(),
    resolved_at     TIMESTAMP,
    notes           TEXT
);

-- Users (role-based access)
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(100) UNIQUE NOT NULL,
    email           VARCHAR(200) UNIQUE NOT NULL,
    password_hash   VARCHAR(200) NOT NULL,
    role            VARCHAR(30) NOT NULL,  -- MUNICIPAL, ADMIN, CITIZEN
    city            VARCHAR(100),
    department      VARCHAR(100),
    active          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    last_login      TIMESTAMP
);

-- AI Reports
CREATE TABLE ai_reports (
    id              BIGSERIAL PRIMARY KEY,
    ward_id         INTEGER REFERENCES wards(id),
    report_text     TEXT NOT NULL,
    model_used      VARCHAR(50),
    generated_at    TIMESTAMP DEFAULT NOW(),
    score_at_time   INTEGER,
    risk_at_time    VARCHAR(20)
);

-- ─── VIEWS ──────────────────────────────────────────────────

-- Latest scores per ward (materialized for performance)
CREATE MATERIALIZED VIEW latest_ward_scores AS
SELECT DISTINCT ON (fs.ward_id)
    w.id,
    w.name,
    w.city,
    ST_Y(w.centroid::geometry) AS lat,
    ST_X(w.centroid::geometry) AS lng,
    w.population,
    fs.score,
    fs.risk_class,
    fs.rainfall_mm,
    fs.elevation_m,
    fs.drainage_pct,
    fs.flood_events_5yr,
    fs.ml_probability,
    fs.computed_at
FROM wards w
JOIN flood_scores fs ON w.id = fs.ward_id
ORDER BY fs.ward_id, fs.computed_at DESC;

CREATE UNIQUE INDEX ON latest_ward_scores(id);

-- City summary view
CREATE VIEW city_flood_summary AS
SELECT
    city,
    COUNT(*) AS total_wards,
    COUNT(*) FILTER (WHERE risk_class = 'RED_ALERT')   AS red_alert,
    COUNT(*) FILTER (WHERE risk_class = 'WATCH_ZONE')  AS watch_zone,
    COUNT(*) FILTER (WHERE risk_class = 'SAFE_ZONE')   AS safe_zone,
    ROUND(AVG(score), 1) AS avg_readiness_score,
    ROUND(AVG(rainfall_mm), 1) AS avg_rainfall_mm,
    MAX(score) AS max_score,
    MIN(score) AS min_score
FROM latest_ward_scores
GROUP BY city
ORDER BY avg_readiness_score ASC;

-- ─── SEED DATA ──────────────────────────────────────────────

INSERT INTO cities (name, state, centroid, population) VALUES
('Mumbai',  'Maharashtra', ST_SetSRID(ST_MakePoint(72.877, 19.076), 4326)::geography, 20667656),
('Pune',    'Maharashtra', ST_SetSRID(ST_MakePoint(73.856, 18.520), 4326)::geography, 6629347),
('Delhi',   'NCT Delhi',   ST_SetSRID(ST_MakePoint(77.102, 28.704), 4326)::geography, 32941309),
('Chennai', 'Tamil Nadu',  ST_SetSRID(ST_MakePoint(80.270, 13.083), 4326)::geography, 10971108),
('Kolkata', 'West Bengal', ST_SetSRID(ST_MakePoint(88.364, 22.573), 4326)::geography, 14850066);

-- Sample ward: Dharavi, Mumbai
INSERT INTO wards (name, city, ward_number, centroid, population) VALUES
('Dharavi',    'Mumbai',  'W01', ST_SetSRID(ST_MakePoint(72.8521, 19.0422), 4326)::geography, 850000),
('Kurla',      'Mumbai',  'W02', ST_SetSRID(ST_MakePoint(72.8794, 19.0726), 4326)::geography, 420000),
('Velachery',  'Chennai', 'C01', ST_SetSRID(ST_MakePoint(80.2209, 12.9785), 4326)::geography, 210000),
('Tondiarpet', 'Chennai', 'C02', ST_SetSRID(ST_MakePoint(80.2889, 13.1227), 4326)::geography, 200000),
('Tiljala',    'Kolkata', 'K01', ST_SetSRID(ST_MakePoint(88.3980, 22.5382), 4326)::geography, 195000);

-- Initial flood scores for seeded wards
INSERT INTO flood_scores (ward_id, score, risk_class, rainfall_mm, elevation_m, drainage_pct, flood_events_5yr)
VALUES
(1, 18, 'RED_ALERT',   210, 8,  25, 4),
(2, 32, 'RED_ALERT',   195, 12, 35, 3),
(3, 15, 'RED_ALERT',   210, 3,  15, 6),
(4, 20, 'RED_ALERT',   188, 4,  18, 5),
(5, 22, 'RED_ALERT',   215, 6,  20, 5);
