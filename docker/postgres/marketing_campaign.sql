CREATE SCHEMA marketing_campaign;

CREATE TABLE marketing_campaign.clients (
    client_id SERIAL PRIMARY KEY,
    age INTEGER,
    job VARCHAR(50),
    marital VARCHAR(50),
    education VARCHAR(50),
    "default" BOOLEAN,
    balance INTEGER,
    housing BOOLEAN,
    loan BOOLEAN,
    contact VARCHAR(50),
    day INTEGER,
    month VARCHAR(50),
    duration INTEGER,
    campaign INTEGER,
    y BOOLEAN
);

CREATE TABLE marketing_campaign.campaigns (
    contact_id SERIAL PRIMARY KEY,
    client_id INTEGER REFERENCES marketing_campaign.clients(client_id),
    contact_date DATE,
    contact_outcome VARCHAR(256)
);
