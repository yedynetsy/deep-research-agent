from tools.reader import read_webpage


url = "https://example.com"

result = read_webpage(url)

print("Title:", result["title"])
print("URL:", result["url"])
print("Content:")
print(result["content"][:1000])
