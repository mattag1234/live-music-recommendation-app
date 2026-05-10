# Lab_9

Run the following upon install:

First start by creating a Virtual Environment 

`python3 -m venv .venv`

Activate it: 

MACOS: 

`source .venv/bin/activate`

WINDOWS: 
`cd C:\path\to\your\project`
`.venv\Scripts\activate`

Once its activated, we can then get the dependencies:

pip install -r requirements.txt 

To start local web server:
1. Open integrated terminal for index.html.
2. Run: python -m http.server
3. Find the webpage at http://localhost:8000 (Ctrl + C to close server)