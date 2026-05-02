import os
from openai import OpenAI
import gradio as gr
import uuid
import chromadb
from pprint import pprint
import json
import requests
import random

#-------------------------------
#Setup
#-------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API key is missing")
client = OpenAI()

MODEL = "gpt-4.1-mini"

#-------------------------------
#Documents
#-------------------------------

context_professional = """
Here is a link to Rachel's linkedin: https://www.linkedin.com/in/rachelphang/

Rachel has leveraged her background in financial services with data skills in the ever-exciting and ever-changing field of
data analytics. She enjoys the technical challenge as well as the reward of seeing something she has built being used in an impactful way.

In the future, she is keen to continue to keep up with the field, building up technical knowledge, as well as building management skills.

What ties all these things together (alongside what is listed in her personal profile) is a keenness to get out of her comfort zone, driving a growth mindset.

RELEVANT EXPERIENCE
Organisation: Citi 
Tenure: Jun 2024 - May 2025
Title: Loans Transformation Data Analyst (Contract)
Geography: London, United Kingdom
Summary: Delivered high-impact data remediation under Citi’s global Lending Transformation Programme, automating regulatory reporting and strengthening compliance.
Partnered with cross-functional teams to optimise data sourcing, enhance governance, and improve risk transparency for senior stakeholders.
Applied advanced workflow automation (KNIME) to replace manual processes, reinforcing a foundation for scalable AI and data-driven solutions.

Organisation: LSEG Capital Markets (formerly Refinitiv)
Tenure: Mar 2019 - Mar 2024 
Geography: London, United Kingdom
Title: FX Transactions Sales Data Analyst
Summary: Built and deployed a suite of Tableau dashboards, transforming sales reporting into real-time insights for global FX teams.
Automated complex manual workflows in Alteryx, saving 90-180 hours per month and pioneering analytics tools that became client-facing products.
Re-engineered SQL pipelines to improve scalability and performance, supporting long-term product and data innovation.
Acted as the sole technical partner to senior sales leadership, gathering requirements and shaping analytics strategy.

Organisation: The Information Lab
Tenure: Feb 2017 - Jan 2019
Geography: London, United Kingdom
Title: Consultant Data Analyst (Tableau and Alteryx)
Summary: Delivered data strategy and analytics solutions for top-tier clients (JP Morgan, UBS, ISS), shaping operating models for scalable data use.
Supported clients in defining data roadmaps and business cases, highlighting ROI of automation and self-service analytics.
Designed and implemented reusable analytics frameworks that accelerated client delivery while embedding governance.
Facilitated stakeholder workshops and training sessions to foster a data-driven culture and sustainable adoption.

EARLY CAREER HIGHLIGHTS
Organisation: Maybank Asset Management
Title: Fixed Income Fund Manager / Credit Analyst
Tenure: Oct 2011 - Jun 2015
Geography: Kuala Lumpur, Malaysia
Summary: Managed long-only fixed income funds for institutional funds, conventional corporate mandates, foundation & retirement (balanced) funds, and high net worth individuals’ portfolios. Performed credit analysis of Asian Corporates and Financial Institutions and trade execution for Asian USD funds as well as local currency (MYR) funds and mandates.

Organisation: Alliance Investment Management
Tenure: Dec 2008 - Sep 2011 · 2 yrs 10 mos
Title: AVP Credit Analyst and CLO Portfolio Manager
Geography: Kuala Lumpur, Malaysia
Summary: Credit analyst supporting fixed income fund manager.
Monitored and managed a RM800 million CLO portfolio of 25 Malaysian performing and non-performing loans through to maturity.
Undertook a lead role in formulating the Alliance Financial Group five year strategy reporting directly to Group CEO as part of the CEO's Special Projects Team, having been selected to handle EVA modelling due to strong financial statement analysis background.

Organisation: Fitch Ratings
Tenure: Aug 2006 - Feb 2008 · 1 yr 7 mos
Title: CDO Performance Analyst (Promoted from CDO Performance Coordinator after 12 months)
Geography: London, United Kingdom
Summary: Developed broad knowledge of the European CDO market and extensive knowledge of various CDO structures, including both cash and synthetic structures, SME CDOs, CLOs, CDO-squared, and market value CDOs, as well as SIVs.
ECA International logo

Organisation: ECA International
Tenure: Feb 2006 - Jul 2006
Title: Cost of Living Research Analyst
Geography: London, United Kingdom
Summary: Research analysis within the Cost of Living department, including inflation research and analysis using in-house systems and developing Excel spreadsheets to facilitate in-depth research of cost of living indicators.  Key statistics kept current and published on a regular basis, as well as on demand to clients.

CERTIFICATIONS, & TECHNICAL SKILLS
Google Cloud Data Analytics Certificate (In Progress, expected completion 2026): Completed 2 of 5 courses; focusing on cloud-native data services, architecture, and leveraging GCP for advanced analytics and reporting.
dbt Fundamentals (2025)
Tableau Desktop Certified Associate (2019)
Alteryx Designer Advanced (2018, 2024) | Alteryx Certified Partner (2017)
KNIME, SQL, Excel/VBA
Data visualization and reporting for compliance, risk, and performance
Financial and risk analysis
AI Engineering
"""

context_personal="""Rachel spent 9 years of her childhood in Jakarta. She also spent 6 years of her childhood in Kuala Lumpur. She has also spent 3 months in Bogota as an adult, living in the La Candelaria area and using it as a base for exploring Colombia. Outside of work, she enjoys exploring new places, both in London (where she lives) and in the wider UK, as well as short haul city breaks in Europe and, when she gets the chance, time to explore further in the rest of the world. To help with that, she also enjoys learning and practicing different natural languages and keeping fit. She spent a few years learning capoeira (although she still isn't very good at it), and loves a good roda."""

context_languages="""Aside from English, Rachel speaks conversational Spanish, Bahasa Indonesia, and Bahasa Malaysia. She has a basic understanding of Cantonese. """

context_food="""Rachel enjoys pasta, noodles, and rice. She enjoys Mexican, Italian, Vietnamese, Nigerian, Malaysian, and Indonesian food, as well as exploring the diverse regional cuisines of China."""

context_education="""Rachel has a
MSc (with Distinction) in Finance and Management, Keele University, UK and a 
BSc (Hons) Economics, Econometrics & Finance, University of York, UK"""

context_dt=f"""
This is how I built my Digital Twin, that you are currently interacting with:

After 5 weeks of hands-on learning in the @SuperDataScience AI Engineering challenge, I deployed a fully functional AI assistant that:

✅ Answers questions about my background
✅ Uses RAG to retrieve relevant info
✅ Via tool-calling can notify me when someone wants to connect

What I learned in the process:
- Prompt engineering (system vs user prompts)
- Tokenization and API Cost management
- Conversation history & context management
- Building chat UIs with Gradio
- RAG (chunking, embeddings, vector stores)
- LLM tool calling (parallel & sequential calls)
- Deploying to Hugging Face Spaces

The model in use is {MODEL}
"""

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
	examples=["Tell me about yourself", "Help me make a decision (roll dice if close)", "Collaborate/reach out to real Rachel"]
).launch()
