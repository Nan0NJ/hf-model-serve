from model_server.schemas import ChatCompletionResponse, Choice, ChoiceMessage, Usage


def test_chat_completion_response_schema():
    response = ChatCompletionResponse(
        id="chatcmpl-test", created=123, model="fake",
        choices=[Choice(message=ChoiceMessage(content="hello"), finish_reason="stop")],
        usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
    ).model_dump()
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert response["usage"]["total_tokens"] == 3
