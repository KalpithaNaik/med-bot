import requests

url = "http://127.0.0.1:5001/chat"
data = {"text": "Tell me a joke about computers."}
  # Change 'message' to 'text'
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        print("Response from server:", response.json())
    else:
        print("Failed to get response. Status code:", response.status_code)
        print("Response text:", response.text)

except requests.exceptions.RequestException as e:
    print("Error sending request:", e)
