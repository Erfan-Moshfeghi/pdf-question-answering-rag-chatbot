from src.answer_generator import AnswerGenerator


def test_answer_generator_returns_insufficient_message_for_low_score_result():
    answerer = AnswerGenerator(min_relevance_score=0.15)

    answer, sources = answerer.generate(
        "What is the capital of France?",
        [
            {
                "text": "The project allows users to upload PDF files and ask questions about their content.",
                "document": "sample.pdf",
                "page_number": 1,
                "chunk_id": 1,
                "score": 0.0,
            }
        ],
    )

    assert answer == "I could not find enough information in the uploaded document to answer this question."
    assert sources == []
