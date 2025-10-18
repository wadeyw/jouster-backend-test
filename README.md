# Jouster Backend

A FastAPI application for text analysis using OpenRouter LLM. Accepts unstructured text, analyzes it, extracts insights, and stores results in PostgreSQL.

**Live Demo:** https://jouster-backend-test.onrender.com

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenRouter API key

### Setup
1. Clone and navigate to the project:
   ```bash
   git clone <repository-url>
   cd jouster-backend
   ```

2. Set environment variables in `docker-compose.yml`:
   ```yaml
   DATABASE_URL=postgresql://user:password@db:5432/jouster
   OPENROUTER_API_KEY=your_api_key_here
   ```

3. Run the application:
   ```bash
   docker-compose up --build
   ```

4. Access at: http://0.0.0.0:8000

> **Note:** Database tables are created automatically on first run.


## API Endpoints

### `POST /analyze`
Analyzes unstructured text using OpenRouter LLM.

**Request:**
```json
{ "text": "Your unstructured text here..." }
```

**Response:** Analysis record with `id`, `created_date`, `summary`, `title`, `topics`, `sentiment`, `keywords`

### `GET /search?topic=<topic>`
Search records by topic.

### `GET /records`
Retrieve all analysis records.

> **Note:** All endpoints validate input and return 400 for empty/whitespace-only values.

## Testing

```bash
docker run --rm jouster-backend-test python -m pytest tests/ -v
```

## Tech Stack

- **FastAPI** - Web framework
- **OpenRouter** - LLM API (deepseek/deepseek-chat-v3.1:free)
- **PostgreSQL** - Database
- **NLTK** - Text processing
- **Docker** - Containerization
- **Render** - Host
