from groq import APIStatusError, RateLimitError

from agent.core import run_agent


def main():
    print("Deep Research Agent")

    query = input("\nEnter your research question:\n> ")

    try:
        report = run_agent(query)

        print("\nResearch Report\n")
        print(report)

    except RateLimitError:
        print("\nGroq rate limit exceeded. Try again after the limit resets.")

    except APIStatusError as error:
        if error.status_code == 413:
            print("\nGroq rejected the request due to token limits.")
            print("Try a shorter research question or fewer sources.")
        else:
            print(f"\nGroq API error: HTTP {error.status_code}")


if __name__ == "__main__":
    main()
