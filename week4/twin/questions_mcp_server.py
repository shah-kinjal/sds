from mcp.server.fastmcp import FastMCP
import questions

mcp = FastMCP("questions_server")


@mcp.tool()
async def get_questions_with_answer() -> str:
    """
    Retrieve from the database all the recorded questions where you have been provided with an official answer.

    Returns:
        A string containing the questions with their official answers.
    """
    return questions.get_questions_with_answer()

@mcp.tool()
async def get_questions_with_no_answer() -> str:
    """
    Retrieve from the database all the recorded questions that dont have an official answer.

    Returns:
        A string containing the questions that have not been answered.
    """
    return questions.get_questions_with_no_answer()

@mcp.tool()
async def record_question_with_no_answer(question: str) -> str:
    """
    Record a question that is not yet recored in the database as a question that does not have an answer.
    Args:
        question: The question that was asked that you couldn't answer
    Returns:
        A string containing the question that was just recoreded. 
    """
    return questions.record_question_with_no_answer(question)

@mcp.tool()
async def record_answer_to_question(id: int, answer: str) -> str:
    """
    Record an answer to a given question.

    Args:
        id: The id of the question that needs an answer
        answer: The answer to the question

    Returns:
        A string containing the question that has been answered and the answer that has been recorded.
    """
    return questions.record_answer_to_question(id, answer)


if __name__ == "__main__":
    mcp.run(transport="stdio")
