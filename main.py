import argparse
from transllm.core import LLMtranslator


def main():
    parser = argparse.ArgumentParser(description="LLM Model Chat")
    parser.add_argument("--hfmodel", type=str, help="Path to the LLM model")
    parser.add_argument(
        "--lang", type=str, default="ko", help="language for translation (default: ko)",
    )
    parser.add_argument(
        "--translator", type=str, default="google", help="translate service API",
    )
    args = parser.parse_args()

    model = LLMtranslator(args.hfmodel, target_lang=args.lang, translator="google")

    print("=== LLM Model Chat ===")
    print("You can start chatting with the LLM model. Enter 'q' to quit.")
    print("-----------------------------------------")

    while True:
        user_input = input("User: ")

        if user_input.lower() == "q":
            print("Exiting the chat.")
            break

        answer = model.generate(user_input)
        print("LLM: ", answer)
        print("-----------------------------------------")


if __name__ == "__main__":
    main()
