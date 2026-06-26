from mcp.server.fastmcp import FastMCP
from .indexer import Indexer
from .plugins.text_plugin import TextPlugin

# Initialize the MCP server
mcp = FastMCP("LanceDB Indexer")

# Initialize the Indexer and add the example plugin
indexer = Indexer()
indexer.add_plugin(TextPlugin())

@mcp.tool()
def index_directory(path: str) -> str:
    """
    Indexes all supported files in the given directory.
    """
    try:
        indexer.index_directory(path)
        return f"Successfully indexed directory: {path}"
    except Exception as e:
        return f"Error indexing directory: {str(e)}"

@mcp.tool()
def search_documents(query: str, limit: int = 5) -> str:
    """
    Search for relevant document chunks using a natural language query.
    """
    try:
        results = indexer.search(query, limit)
        if not results:
            return "No results found."
        
        output = []
        for res in results:
            output.append(f"Title: {res['title']}\nDoc ID: {res['doc_id']}\nText: {res['text']}\n---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

@mcp.tool()
def fetch_document(doc_id: str) -> str:
    """
    Fetch all chunks of a document by its ID.
    """
    try:
        results = indexer.fetch(doc_id)
        if not results:
            return "Document not found."
        
        output = []
        for res in results:
            output.append(res['text'])
        return "\n\n".join(output)
    except Exception as e:
        return f"Error fetching document: {str(e)}"

if __name__ == "__main__":
    mcp.run()
