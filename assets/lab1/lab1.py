import sys
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


def get_response(input: str, client: OpenAI):
    response = client.responses.create(model="gpt-5", input=input)
    return response


if __name__ == "__main__":
    print("Введите ваш вопрос (или 'exit' для завершения):")
    while True:
        question = input("Вы: ")
        if question.lower() == "exit":
            print("Завершение программы.")
            break
        answer = get_response(question, client)
        print("AI:", answer.output_text)
