import sqlite3
from pathlib import Path

db_path = Path("mytwin") / Path("memory") / Path("questions.db")
DB = db_path.absolute()


def record_question_in_db(question: str) -> str:
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO questions (question, answer) VALUES (?, NULL)", (question,))
        conn.commit()
        return "Recorded question with no answer"


def fetch_questions_from_db_with_no_answer() -> str:
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, question FROM questions WHERE answer IS NULL")
        rows = cursor.fetchall()
        if rows:
            return "\n".join(f"Question id {row[0]}: {row[1]}" for row in rows)
        else:
            return "No questions with no answer found"


def fetch_questions_from_db_with_answer() -> str:
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer FROM questions WHERE answer IS NOT NULL")
        rows = cursor.fetchall()
        return "\n".join(f"Question: {row[0]}\nAnswer: {row[1]}\n" for row in rows)


def record_or_update_answer_to_question(id: int, answer: str) -> str:
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE questions SET answer = ? WHERE id = ?", (answer, id))
        conn.commit()
        return "Recorded answer to question"

def fetch_answer_for_question(id: int) -> str:
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT answer FROM questions WHERE id = ?", (id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            return "No answer found for this question"
