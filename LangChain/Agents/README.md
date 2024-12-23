<h1>LangChain Agents using HuggingFace model</h1>

<p>
    <b>
    Objective: Familiar with Agents procedure in LangChain using a HuggingFace model.
    </b>
</p>
<h2>Guilds:</h2>
<ul>
    <li><a href="https://python.langchain.com/docs/tutorials/agents/">LangChain Agents</a></li>
    <li><a href="https://python.langchain.com/docs/concepts/agents/">Agents</a></li>
    <li><a href="https://python.langchain.com/docs/concepts/chat_models/">LangChain Chat model</a></li>
    <li><a href="https://python.langchain.com/docs/integrations/providers/huggingface/">LangChain and HuggingFace</a></li>
</ul>
<h2>There are some steps to build Agents in LangChain:</h2>

<h3>1. Define tools</h3>
<ul>
    <li>Main tool of choice will be Tavily - a search engine</li>
</ul>
<h3>2. Define a LLM ChatModel from HuggingFace checkpoint</h3>
<ul>
    <li>Check the chat prompt templates of each model</li>
</ul>
<h3>3. Enable this model to do tool calling</h3>
<ul>
    <li>Use .bind_tools to give the language model knowledge of these tools</li>
</ul>
<h3>4. Create the Agent</h3>
<ul>
    <li>4.1. Initialize the agent</li>
    <li>4.2. Run the agent</li>
</ul>
<h3>5. Additional Improvements for Agents</h3>
<ul>
    <li>5.1. Streaming messages</li>
    <li>5.2. Streaming tokens</li>
    <li>5.3. Adding in memory</li>
</ul>