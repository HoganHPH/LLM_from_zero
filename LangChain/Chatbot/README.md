<h1>Chatbot in LangChain using HuggingFace model</h1>

<p><b>Note*: Define prompt template in LangChain when using a HuggingFace model: <a href='https://stackoverflow.com/questions/76178954/giving-systemmessage-context-to-conversationalretrievalchain-and-conversationbuf'>LangChain promt_template with HuggingFace</a></b></p>

<p>
    <b>
    Objective: Familiar with Chatbot procedure in LangChain using a HuggingFace model.
    </b>
</p>
<h2>Guilds:</h2>
<ul>
    <li><a href="https://python.langchain.com/docs/tutorials/chatbot/">LangChain Chatbot</a></li>
    <li><a href="https://python.langchain.com/docs/concepts/chat_models/">LangChain Chat model</a></li>
    <li><a href="https://python.langchain.com/docs/integrations/providers/huggingface/">LangChain and HuggingFace</a></li>
</ul>
<h2>There are many ways to build a Chatbot in LangChain:</h2>

<h3>1. Basic Chatbot</h3>
<ul>
    Procedures:
    <li>Step 1: Define a LLM ChatModel using pretrained HuggingFace checkpoint</li>
    <li>Step 2: Use model to invoke a message</li>
</ul>
<p><b>Note*: Basic Chatbot fails in remembering conversation history</b></p>
<h3>2. Chatbot with message persistence</h3>
<ul>
    Procedures:
    <li>Step 1: Define a LLM ChatModel using pretrained HuggingFace checkpoint</li>
    <li>Step 2: Define chatbot app using memory workflow of LangGraph</li>
    <li>Step 3: Define configuration to identify the speaker (Who is speaking?). Note*: have to define and pass into every time</li>
    <li>Step 4: Use defined chatbot app to invoke human message</li>
    <li>Addition: async function</li>
</ul>
<h3>3. Chatbot with prompt templates</h3>
<ul>
    <li>Step 1: Define a LLM ChatModel using pretrained HuggingFace checkpoint</li>
    <li>Step 2: Define prompt template or Define an instance containing variables for prompt</li>
    <li>Step 3: Init workflow of LangGraph</li>
    <li>Step 4: Pass into instructions for prompt and Invoke message</li>
</ul>
<h3>4. Chatbot with managing conversation history</h3>
<ul>
    <li>Step 1: Define a LLM ChatModel using pretrained HuggingFace checkpoint</li>
    <li>Step 2: Define prompt template</li>
    <li>Step 3: Define an instance containing variables for prompt</li>
    <li>Step 4: Define trimmer for saving conversation history</li>
    <li>Step 5: Define workflow of LangGraph</li>
    <li>Step 6: Define config and Invoke message</li>
</ul>
<h3>5. Chatbot with streaming</h3>
<ul>
    <li>Step 1: Define a LLM ChatModel using pretrained HuggingFace checkpoint</li>
    <li>Step 2: Define prompt template</li>
    <li>Step 3: Define an instance containing variables for prompt</li>
    <li>Step 4: Define workflow of LangGraph</li>
    <li>Step 5: stream back each token as it is generated</li>
</ul>





