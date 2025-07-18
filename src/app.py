import sys
import pandas as pd
from llm_simplifier import simplify_math_text

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)
        df["Simplified"] = df["Text"].apply(simplify_math_text)
        output_path = "simplified_output.csv"
        df.to_csv(output_path, index=False)
        print(f"Simplified results saved to {output_path}")
    else:
        print("Welcome to the Math Text Simplifier!")
        print("Type 'exit' or press Ctrl+C to quit.\n")
        
        while True:
            user_input = input("Enter a math sentence to simplify: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            result = simplify_math_text(user_input)
            print("\nSimplified version:\n", result, "\n")

if __name__ == "__main__":
    main()
