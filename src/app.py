from llm_simplifier import simplify_math_text

def main():
    print("Welcome to the Math Text Simplifier!")
    user_input = input("Simplify this math text: ")
    result = simplify_math_text(user_input)
    print("\nSimplified version:\n", result)

if __name__ == "__main__":
    main()
