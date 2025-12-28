from mcp.server.fastmcp import FastMCP
import db_tools

mcp = FastMCP("dbtoos_server")


@mcp.tool()
async def get_questions_with_answer() -> str:
    """
    Retrieve from the database all the recorded questions where you have been provided with an official answer.

    Returns:
        A string containing the questions with their official answers.
    """
    return db_tools.fetch_questions_from_db_with_answer()

@mcp.tool()
async def get_questions_with_no_answer() -> str:
    """
    Retrieve from the database all the recorded questions that dont have an official answer.

    Returns:
        A string containing the questions that have not been answered.
    """
    return db_tools.fetch_questions_from_db_with_no_answer()

@mcp.tool()
async def record_question_in_db(question: str) -> str:
    """
    Record a question that is not yet recored in the database as a question that does not have an answer.
    Args:
        question: The question that was asked that you couldn't answer
    Returns:
        A string containing the question that was just recoreded. 
    """
    return db_tools.record_question_in_db(question)

@mcp.tool()
async def record_answer_to_question(id: int, answer: str) -> str:
    """
    Record or update an answer to a given question.

    Args:
        id: The id of the question that needs an answer
        answer: The answer to the question

    Returns:
        A string containing the question that has been answered and the answer that has been recorded.
    """
    return db_tools.record_or_update_answer_to_question(id, answer)

@mcp.tool()
async def get_answer_to_question(id: int) -> str:
    """
    Fetch the answer for a given question.
    Args:
        id: The id of the question that needs an answer
    Returns:
        A string containing the question and the answer to the question.
    """
    return db_tools.fetch_answer_for_question(id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
