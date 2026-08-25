"""
Terminal chatbot for the school data assistant.
Run this while both MCP servers (MongoDB on :8002, MySQL on :8001) are up.
"""
 
import asyncio
 
import mcp_manager
from agent import run_agent
from config import SYSTEM_PROMPT
 
 
async def main():
    print("Connecting to data sources...")
    ollama_tools = await mcp_manager.discover_tools()
 
    if not ollama_tools:
        print("No tools discovered from either server. Check that both MCP "
              "servers are running and reachable, then try again.")
        return
 
    print(f"Connected. {len(ollama_tools)} tool(s) available:")
    for t in ollama_tools:
        print(f"  - {t['function']['name']}")
 
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    print("\nSchool Data Assistant — ask a question, or type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue
 
        answer, _ = await run_agent(question, ollama_tools, history)
        print(f"\nAssistant: {answer}\n")
 
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
 
 
if __name__ == "__main__":
    asyncio.run(main())
 
 