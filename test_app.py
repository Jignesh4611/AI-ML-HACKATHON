import requests 
url= "http://127.0.0.1:5000/analyze"
response= requests.post(url,json={"text":"i love this project"})
print(response.json())