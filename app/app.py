import os
from openai import OpenAI
import gradio as gr
import uuid
import chromadb
from pprint import pprint
import json
import requests
import random
from huggingface_hub import hf_hub_download

#-------------------------------
#Setup
#-------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API key is missing")
client = OpenAI()

MODEL = "gpt-4.1-mini"

HF_TOKEN=os.getenv("HF_TOKEN")

#-------------------------------
#Documents
#-------------------------------

doc_files={
"Professional" : "context_professional.txt",
"Personal": "context_personal.txt",
"Languages": "context_lang.txt",
"Food": "context_food.txt",
"Education": "context_educ.txt",
"Certifications and Skills": "context_certs_and_skills.txt",
"About the digital twin": "context_dt.txt"
}

documents=[]

for source_name, filename in doc_files.items():
	path=hf_hub_download(repo_id="datarachel/profile",
	filename=filename,
	repo_type="dataset",
	token=HF_TOKEN)

with open(path "r") as f:
	text = f.read()
	documents.append({"text":text, "source":source_name})
	print(f"Loaded:{source_name}")

#-------------------------------
#Chunking Function
#-------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        # If the remaining text fits in one chunk, we're done.
        if end >= len(text):
            chunks.append(text[start:])
            break

        window = text[start:end]
        midpoint = chunk_size // 2
        # Try each boundary type from most to least preferred.
        cut = None
        for boundary in ["\n\n", "\n", ". ", "! ", "? ", " "]:
            pos = window.rfind(boundary, midpoint)
            if pos != -1:
                cut = pos + len(boundary)  # cut after the boundary characters
                break

        if cut is None:
            cut = chunk_size  # no boundary found; hard cut at the limit
        chunks.append(text[start : start + cut])
        # Next chunk starts `overlap` characters before the cut point.
        next_start = start + cut - overlap
        # Ensure we always move forward to avoid infinite loops.
        start = max(next_start, start + 1)

    return chunks

#-------------------------------
#RAG: Chunk, Embed & Store in ChromaDB
#-------------------------------

documents = [
    {"text": context_professional, "source": "context_professional"},
    {"text": context_personal, "source": "context_personal"},
    {"text": context_languages, "source": "context_languages"},
    {"text": context_education, "source": "context_education"},
    {"text": context_food, "source": "context_food"},
    {"text": context_dt, "source": "context_dt"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    #Prepare the lists
    chunks_ = chunk_text(doc["text"], 300, 30)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks_))]
    #Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

#print for logs
print(f"Created {len(chunks)} chunks:\n")

for i, chunk in enumerate(chunks):
    print(f"--- chunk {i+1} | {len(chunk)} chars --- (ID: {ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}):")
    print(chunk)
    print()

#Generate embeddings for all chunks
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)
embeddings = [item.embedding for item in response.data]

#Verify embeddings for logs
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

#initialise ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db_twin")
#Alternative: initalise ChromaDB client (in-memory storage)
#chroma_client = chromadb.Client()

#Get or Create + Empty the collection before adding new data (for testing purposes)
collection = chroma_client.get_or_create_collection(name="RP_digital_twin")
if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

#Adding data to ChromaDB
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)

pprint(collection.get())

#-------------------------------
#Tools
#-------------------------------
tools = []

#Pushover
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

#Create send notification function
def send_notification(message: str):
	if pushover_user is None or pushover_token is None: #Handling of missing credentials
		return "Notification failed: Pushover not configured."
	payload = {"user": pushover_user, "token": pushover_token, "message": message}
	requests.post(pushover_url, data=payload)
	return f"Notification sent: {message}"

#Describe Pushover as an LLM Tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notification to real-world version of you via Pushover on mobile. Use this when:\
    1) the user needs to alert the real-world version version of you, \
    2) the user wants to get in touch, hire, or collaborate. Ask them for their name and contact details first, then send this information over to\
    real-world Rachel in a notification. Follow this up requesting how real world Rachel might be able to help them.\
    3) you don't know the answer to a question that the user has asked about Rachel - send automatically without asking. Include the question so that she\
    can add this information later.",
    "parameters": {
        "type": "object",
        "properties": {
           "message": {"type": "string", "description": "The notification message to send to the user's device"} 
        },
        "required": ["message"]
    }
}

# Add Pushover to the list of tools for the LLM
tools.append({"type": "function", "function":send_notification_function})

#Simulates rolling a single six-sided die and returns the result
def dice_roll():
    result = random.randint(1,6)
    return result

#Describe dice roll function for the LLM
roll_dice_function = {
    "name": "dice_roll",
    "description": "Simulates rolling a six-sided die and returns the result. Use this when the user wants to roll a die for games, decisions, or random number generation.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    },
}

#add function to the list tools available to LLM

tools.append({"type": "function", "function":roll_dice_function})

#-------------------------------
#Tool Handler
#-------------------------------
def handle_tool_call(tool_calls):
    tool_results = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        #print(f"Calling function {function_name}") #for future debugging

        #Route to the appropriate function based on function_name
        if function_name == "send_notification":
            content = send_notification(args["message"])
        elif function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"
        else:
            content = f"Unknown function: {function_name}"

        tool_call_result = {
        	"role": "tool",
        	"content": content,
        	"tool_call_id": tool_call.id
        }
        tool_results.append(tool_call_result)

    #print("Final message:", message)
    return tool_results
#-------------------------------
#System Message
#-------------------------------

system_message = """ You are a digital twin of Rachel Phang that\
answers questions based on the provided context. When people talk to you, \
you respond as Rachel - in first person, using her voice and knowledge. \
Speak with a dynamic, helpful, confident and competent, yet slightly sassy, tone.\
Please answer as completely and as exhaustively, yet conscisely and elegantly, as possible.\
Use a British or UK-friendly tone and spellings.\
If someone wishes to converse with you in a different language, feel free to continue the conversation in that language.\
IMPORTANT: If you don't know the answer based on the context, say you don't know. Always use all available information to provide \
the best answer possible, but please absolutely do not make anything up. \
The only factual information available to you is what is in this system message.\
You cannot get any more facts about Rachel from the internet or make them up.\
IMPORTANT: Whenever you don't know something about Rachel, ALWAYS use the send_notification tool to alert the real Rachel.\
Do this automatically without asking or alerting the user.

<output_contract>
- Return exactly the sections requested, in the requested order.
- If the prompt defines a preamble, analysis block, or working section, do not treat it as extra output.
- Apply length limits only to the section they are intended for.
- If a format is required (JSON, Markdown, SQL, XML), output only that format.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Avoid repeating the user's request.
- Keep progress updates brief.
- Do not shorten the answer so aggressively that required evidence, reasoning, or completion checks are omitted.
</verbosity_controls>
"""

#-------------------------------
#Main Response Function
#-------------------------------

def respond_ai(message,history):
    #RAG: Embed the query using the same model we used for the chunks to ensure compatability
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
        #input = [test_query, test_query2]
    )

    query_embedding=response.data[0].embedding
    
    #RAG: Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10
    )

    #RAG: Stitch retrieved chunks together to create the context for the response
    context = "\n---\n".join(results["documents"][0])

    #Print logs for debugging
    print("\n==============================================\n")
    print(f"User message:\n{message}\n")
    print(f"***Retrieved Chunks:")
    for a, b in zip(results["documents"][0], results["metadatas"][0]):
        print("--------------------------------------")
        print(f"Document: {b['source']} --Chunk {b['chunk_index']}>>\n{a}\n")
    
    #Update system message with context (for this conversation turn)
    system_message_enhanced = system_message + "\n\nContext:\n" + context

    #Logs for debugging
    #print("\n==============================================\n")
    #print("***User message:\n", message)
    #print("\n***Context this turn:\n", system_message_enhanced)

    #Verify retrieval works - print which chunks were retrieved and their content
    #pprint(results)
    #print(f"Query: {message}\n")
    #print("Retrieved Chunks:")
    #for a, b, c in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    #    print(f"Chunk distance {c} \n Chunk {b['chunk_index']}:\n{a}\n")

    #Build messages for this turn
    messages = [{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}]
    
    #Call LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
   
    message = response.choices[0].message

    while message.tool_calls:
        pprint(message.tool_calls)

        tool_result = handle_tool_call(message.tool_calls)
        messages.append(message)
        messages.extend(tool_result)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message

        #Note: Maybe consider adding protection from infinite consecutive tool calling
    
    if message.content:
    	return(message.content)
    else:
    	return ""
    
#-------------------------------
#Launch Gradio
#-------------------------------

gr.ChatInterface(
	fn=respond_ai,
	title="Digital Rachel Phang",
	chatbot=gr.Chatbot(avatar_images=(None, "rachel.jpg"), height=600),
	description="Chat with an AI version of Rachel Phang. Ask about her experience, projects, or just say hi. You can also ask her to roll one or more dice!",
	examples=["Tell me about yourself", "Help me make a decision (roll dice if close)", "Collaborate/reach out to real Rachel", "...or just start chatting below!"]
).launch()
