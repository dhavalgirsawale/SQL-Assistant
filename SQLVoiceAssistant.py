import os
import re
import openai
import psycopg2
import speech_recognition as sr
import pyttsx3
from datetime import datetime
from tabulate import tabulate
from dotenv import load_dotenv

# =============================
# Load Environment Variables
# =============================
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"  # fallback to OpenAI

# =============================
# PostgreSQL Config
# =============================
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}


# =============================
# Text-to-Speech Engine
# =============================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def get_voice_command(prompt=None, retries=3):
    r = sr.Recognizer()
    attempt = 0
    while attempt < retries:
        with sr.Microphone() as source:
            if prompt:
                speak(prompt)
            print("\nListening... (Say 'exit' to quit)")
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source, timeout=5)
                command = r.recognize_google(audio).lower()
                print(f"You said: {command}")
                return command
            except sr.UnknownValueError:
                speak("Sorry, I didn't catch that. Please try again.")
                attempt += 1
            except sr.WaitTimeoutError:
                speak("Listening timed out. Please try again.")
                attempt += 1
    speak("Maximum retries reached.")
    return ""

def clean_sql(sql):
    return re.sub(r"```sql|```", "", sql).strip()

def generate_sql_with_openai(command, schema_info):
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a PostgreSQL assistant. Convert natural language to raw SQL queries. "
                    "Do not explain. Only return SQL. Use case-insensitive filters with LOWER(). "
                    f"Schema:\n{schema_info}"
                )
            },
            {"role": "user", "content": command}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1
        )
        sql = response["choices"][0]["message"]["content"].strip()
        return clean_sql(sql)
    except Exception as e:
        speak(f"OpenAI error: {str(e)}")
        return None

def self_heal_sql(sql):
    return re.sub(r"=\s*([a-zA-Z_][a-zA-Z0-9_]*)", r"= '\1'", sql)

def auto_lowercase_where(sql):
    return re.sub(
        r"WHERE\s+(\w+)\s*=\s*'([^']+)'",
        lambda m: f"WHERE LOWER({m.group(1)}) = LOWER('{m.group(2)}')",
        sql,
        flags=re.IGNORECASE
    )

def detect_operation(sql):
    first = sql.strip().lower().split()[0]
    if "create database" in sql.lower():
        return "create_db"
    elif "drop database" in sql.lower():
        return "drop_db"
    elif first in ["select", "insert", "update", "delete", "create", "drop", "alter", "truncate"]:
        return first
    return "other"

def detected_table_name(sql):
    match = re.search(r"from\s+(\w+)", sql.lower())
    return match.group(1) if match else None

def extract_filters(sql):
    filters = {}
    matches = re.findall(r"lower\((\w+)\)\s*=\s*lower\('([^']+)'\)", sql.lower())
    for col, val in matches:
        filters[col] = val
    return filters

def get_schema_info(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name, column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public'
            """)
            schema = {}
            for table, column, dtype in cursor.fetchall():
                if table not in schema:
                    schema[table] = []
                schema[table].append(f"{column} ({dtype})")
            return "\n".join([f"{table}: " + ", ".join(cols) for table, cols in schema.items()])
    except psycopg2.Error as e:
        return f"Schema retrieval error: {e}"

def log_interaction(command, sql, result):
    with open("query_log.txt", "a") as f:
        f.write(f"\n[{datetime.now()}]\nCommand: {command}\nSQL: {sql}\nResult: {result}\n")

def execute_sql(conn, sql, operation, conversation_context):
    try:
        if operation == "select":
            sql = auto_lowercase_where(sql)

        with conn.cursor() as cursor:
            cursor.execute(sql)
            if operation == "select":
                rows = cursor.fetchall()
                headers = [desc[0] for desc in cursor.description]
                if rows:
                    table = tabulate(rows, headers, tablefmt="psql")
                    print("\nResult:\n" + table)
                    conversation_context["last_table"] = detected_table_name(sql)
                    conversation_context["last_filters"] = extract_filters(sql)
                    conversation_context["last_result"] = rows
                    return "Query executed successfully. Data displayed in table format."
                else:
                    return "No results found."
            else:
                conn.commit()
                return f"{operation.capitalize()} executed successfully."
    except Exception as e:
        conn.rollback()
        healed_sql = self_heal_sql(sql)
        if healed_sql != sql:
            speak("Trying auto-healed version of query...")
            return execute_sql(conn, healed_sql, operation, conversation_context)
        return f"Error: {e}"

def fallback_search(conn, user_input, last_table):
    if not last_table:
        speak("No previous table context available for fallback search.")
        return

    try:
        with conn.cursor() as cursor:
            # Securely check for valid table
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = %s
            """, (last_table,))
            if cursor.fetchone() is None:
                speak(f"Table {last_table} not found in the database.")
                return

            # Find text columns
            cursor.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND data_type IN ('character varying', 'text')
            """, (last_table,))
            text_columns = [row[0] for row in cursor.fetchall()]

            if not text_columns:
                speak(f"No text columns found to search in {last_table}.")
                return

            # Dynamic LIKE search
            conditions = " OR ".join([f"{col} ILIKE %s" for col in text_columns])
            query = f"SELECT * FROM {last_table} WHERE {conditions}"
            params = tuple([f"%{user_input}%"] * len(text_columns))

            cursor.execute(query, params)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]

            if rows:
                table = tabulate(rows, headers, tablefmt="psql")
                print("\nFallback Result:\n" + table)
                speak(f"Showing similar results from {last_table}.")
            else:
                speak(f"No similar data found in {last_table}.")
    except Exception as e:
        speak(f"Fallback error: {e}")

def create_database(db_name):
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database="template1"
        )
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {db_name};")
        conn.close()
        return True
    except Exception as e:
        speak(f"Error creating database: {e}")
        return False

def drop_database(db_name):
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database="template1"
        )
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name};")
        conn.close()
        return True
    except Exception as e:
        speak(f"Error dropping database: {e}")
        return False

def main():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        speak(f"Database connection failed: {e}")
        return

    speak("Voice SQL Assistant is now active.")
    schema_info = get_schema_info(conn)

    conversation_context = {
        "last_table": None,
        "last_filters": {},
        "last_result": None
    }

    while True:
        command = get_voice_command()
        if not command or "exit" in command:
            speak("Goodbye!")
            break

        # Pronoun replacement with safer regex
        for pronoun in ["he", "she", "they", "it", "that"]:
            for col, val in conversation_context["last_filters"].items():
                command = re.sub(rf"\b{pronoun}\b", val, command, flags=re.IGNORECASE)

        if "use database" in command:
            db = command.split("use database")[-1].strip().replace(" ", "_")
            DB_CONFIG["database"] = db
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                schema_info = get_schema_info(conn)
                speak(f"Switched to database {db}")
            except Exception as e:
                speak(f"Could not connect to database {db}: {e}")
            continue

        sql = generate_sql_with_openai(command, schema_info)
        if not sql or "<" in sql:
            speak("Your request seems incomplete. Please provide all required details.")
            continue

        print(f"\nGenerated SQL:\n{sql}")
        op = detect_operation(sql)

        if op in ["drop", "delete", "drop_db", "truncate", "update"]:
            speak("This action will modify or delete data. Say YES to confirm.")
            confirm = get_voice_command()
            if "yes" not in confirm:
                speak("Cancelled.")
                continue

        if op == "create_db":
            db_name = command.split("create database")[-1].strip()
            if create_database(db_name):
                speak(f"Database {db_name} created.")
            continue

        if op == "drop_db":
            db_name = command.split("drop database")[-1].strip()
            if drop_database(db_name):
                speak(f"Database {db_name} dropped.")
            continue

        result = execute_sql(conn, sql, op, conversation_context)
        log_interaction(command, sql, result)

        if "no results found" in result.lower() and op == "select":
            fallback_search(conn, command, conversation_context["last_table"])
        else:
            speak(result)

    conn.close()

if __name__ == "__main__":
    main()


