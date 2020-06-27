import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'message':'Some random text'})

print(r.json())