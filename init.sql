-- Database initialization script for Jouster Backend

-- Create the analysis_records table
CREATE TABLE IF NOT EXISTS analysis_records (
    id SERIAL PRIMARY KEY,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    title VARCHAR(255),
    topics TEXT ,
    sentiment VARCHAR(20),
    keywords TEXT
);

