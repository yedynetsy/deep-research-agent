from tools.search import search_web


results = search_web(
    query="retrieval augmented generation hallucination reduction",
    max_results=3,
)

for index, source in enumerate(results, start=1):
    print(f"\nSource {index}")
    print(f"Title: {source['title']}")
    print(f"URL: {source['url']}")
    print(f"Content: {source['content'][:300]}")
