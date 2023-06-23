import argparse
from transllm.core import LLMtranslator
from googletrans import Translator


def main():
    parser = argparse.ArgumentParser(description="LLM Model Chat")
    parser.add_argument("model_path", type=str, help="Path to the LLM model")
    parser.add_argument(
        "--dest",
        type=str,
        default="ko",
        help="Destination language for translation (default: ko)",
    )
    args = parser.parse_args()

    translator = Translator()
    model = LLMtranslator(args.model_path, dest=args.dest, translator=translator)

    print("=== LLM Model Chat ===")
    print("You can start chatting with the LLM model. Enter 'q' to quit.")
    print("-----------------------------------------")

    while True:
        user_input = input("User: ")

        if user_input.lower() == "q":
            print("Exiting the chat.")
            break

        answer = model.get_answer(user_input)
        print("LLM: ", answer)
        print("-----------------------------------------")
