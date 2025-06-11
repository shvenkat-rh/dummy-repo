# down.py

import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python down.py <issue_title> <issue_body>")
        sys.exit(1)

    issue_title = sys.argv[1]
    issue_body = sys.argv[2]

    result = (
        "### 🔍 Issue Information Extracted by `down.py`:\n\n"
        f"**Title:** {issue_title}\n\n"
        f"**Body:** {issue_body}\n"
    )
    print(result)

if __name__ == "__main__":
    main()
