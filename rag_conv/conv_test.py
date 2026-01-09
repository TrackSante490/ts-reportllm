from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def main():
    llm = ChatOllama(model="llama3.1",temperature=0)
    messages = [
            SystemMessage(content=(
            "You are TrackSante's AI measurement coach. You are to ask concise questions "
            "to the user to get more information on their symptoms, health status, and lifestyle habits. "
            "Be empathetic, supportive, and non-judgmental in your responses. "
            "Keep your questions clear and focused on understanding the user's health needs."
        ))
    ]
    
    while(True):
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        messages.append(AIMessage(content=response.content))

        print("AI Measurement Coach:", response.content)

if __name__ == "__main__":
    main()