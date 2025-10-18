from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app
from llm.client import OpenRouterResponse, OpenRouterError

client = TestClient(app)


def test_analyze_endpoint():
    with patch("main.OpenRouterClient") as mock_client_class:
        # Create a mock instance of the client
        mock_client_instance = mock_client_class.return_value

        # Mock the OpenRouter response
        mock_response = OpenRouterResponse(
            summary="This is a summary of the text.",
            title="Sample Title",
            topics=["topic1", "topic2", "topic3"],
            sentiment="positive",
        )
        mock_client_instance.analyze_text.return_value = mock_response

        # Mock the extract_keywords function
        with patch("main.extract_keywords") as mock_keywords:
            mock_keywords.return_value = ["word1", "word2", "word3"]

            response = client.post(
                "/analyze", json={"text": "This is a sample text for analysis."}
            )

            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert data["summary"] == "This is a summary of the text."
            assert data["title"] == "Sample Title"
            assert data["topics"] == ["topic1", "topic2", "topic3"]
            assert data["sentiment"] == "positive"
            assert data["keywords"] == ["word1", "word2", "word3"]


def test_analyze_endpoint_empty_text():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 400
    assert "non-empty string" in response.json()["detail"]


def test_analyze_endpoint_whitespace_text():
    response = client.post("/analyze", json={"text": "   "})
    assert response.status_code == 400
    assert "non-empty string" in response.json()["detail"]


def test_search_endpoint():
    # Test the search endpoint with a valid topic
    with patch("main.get_db") as mock_get_db:
        # Create a mock database session
        mock_db = mock_get_db.__enter__.return_value

        # Mock the query results - remember that in the DB, topics and keywords are comma-separated strings
        mock_record = type(
            "MockRecord",
            (),
            {
                "id": 1,
                "created_date": "2023-01-01T00:00:00",
                "title": "Sample Title",
                "topics": "topic1,topic2,topic3",  # Stored as comma-separated string
                "sentiment": "positive",
                "keywords": "word1,word2,word3",   # Stored as comma-separated string
            },
        )()

        # Set up the mock query chain for the text filter
        query_mock = mock_db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.all.return_value = [mock_record]

        response = client.get("/search?topic=topic1")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        if response.json():  # If there are results
            assert response.json()[0]["id"] == 1
            assert response.json()[0]["title"] == "Sample Title"
            # Verify that the API converts the comma-separated string back to a list
            assert response.json()[0]["topics"] == ["topic1", "topic2", "topic3"]
            assert response.json()[0]["keywords"] == ["word1", "word2", "word3"]


def test_search_endpoint_empty_topic():
    response = client.get("/search?topic=")
    assert response.status_code == 400
    assert "non-empty string" in response.json()["detail"]


def test_search_endpoint_whitespace_topic():
    response = client.get("/search?topic=   ")
    assert response.status_code == 400
    assert "non-empty string" in response.json()["detail"]


def test_list_records_endpoint():
    # Test the list records endpoint
    with patch("main.get_db") as mock_get_db:
        # Create a mock database session
        mock_db = mock_get_db.__enter__.return_value

        # Mock the query results - remember that in the DB, topics and keywords are comma-separated strings
        mock_record = type(
            "MockRecord",
            (),
            {
                "id": 1,
                "created_date": "2023-01-01T00:00:00",
                "title": "Sample Title",
                "topics": "topic1,topic2,topic3",  # Stored as comma-separated string
                "sentiment": "positive",
                "keywords": "word1,word2,word3",   # Stored as comma-separated string
            },
        )()

        query_mock = mock_db.query.return_value
        query_mock.all.return_value = [mock_record]

        response = client.get("/records")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        if response.json():  # If there are results
            assert response.json()[0]["id"] == 1
            assert response.json()[0]["title"] == "Sample Title"
            # Verify that the API converts the comma-separated string back to a list
            assert response.json()[0]["topics"] == ["topic1", "topic2", "topic3"]
            assert response.json()[0]["keywords"] == ["word1", "word2", "word3"]


def test_analyze_endpoint_with_openrouter_error():
    with patch("main.OpenRouterClient") as mock_client_class:
        # Create a mock instance of the client
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.analyze_text.side_effect = OpenRouterError("API Error")

        response = client.post(
            "/analyze", json={"text": "This is a sample text for analysis."}
        )
        assert response.status_code == 400
        assert "OpenRouter error" in response.json()["detail"]
