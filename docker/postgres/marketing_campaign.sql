CREATE SCHEMA marketing_campaign;

CREATE TABLE marketing_campaign.campaigns (
    contact_id SERIAL PRIMARY KEY,
    client_id INTEGER REFERENCES marketing_campaign.clients(client_id) NOT NULL,
    contact_date DATE NOT NULL,
    contact_outcome VARCHAR(256) NOT NULL
);

CREATE TABLE marketing_campaign.clients (
    client_id SERIAL PRIMARY KEY,
    age INTEGER NOT NULL,
    job VARCHAR(50) NOT NULL,
    marital VARCHAR(50) NOT NULL,
    education VARCHAR(50) NOT NULL,
    default BOOLEAN NOT NULL,
    balance INTEGER NOT NULL,
    housing BOOLEAN NOT NULL,
    loan BOOLEAN NOT NULL,
    contact VARCHAR(50) NOT NULL,
    day INTEGER NOT NULL,
    month VARCHAR(50) NOT NULL,
    duration INTEGER NOT NULL,
    campaign INTEGER NOT NULL,
    y BOOLEAN NOT NULL
);
