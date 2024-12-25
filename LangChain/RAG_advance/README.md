<h1>Basic Retrieval Augmented Generation (RAG) - Part 2</h1>

<p>
    <b>
    Objective: Building an advanced RAG mechanism.
    </b>
</p>
<p>
    <ul>
      <b></b>Key technologies:</b>
      <li>Chat model: <a href='https://huggingface.co/microsoft/Phi-3-mini-4k-instruct'>Phi-3-mini-4k-instruct</a> from HuggingFace</li>
      <li>Embedding model: <a href='https://huggingface.co/sentence-transformers/all-mpnet-base-v2'>all-mpnet-base-v2 from HuggingFace</a></li>
      <li>Vector store: MongoDB</li>
    </ul>
</p>

<h2>Guilds:</h2>
<ul>
    <li><a href="https://python.langchain.com/docs/tutorials/qa_chat_history/">LangChain RAG part 2</a></li>
    <li><a href="https://python.langchain.com/docs/tutorials/rag/">LangChain RAG part 1</a></li>
    <li><a href="https://python.langchain.com/docs/concepts/chat_models/">LangChain Chat model</a></li>
    <li><a href="https://python.langchain.com/docs/integrations/vectorstores/">LangChain Vector store</a></li>
    <li><a href="https://python.langchain.com/docs/integrations/providers/huggingface/">LangChain and HuggingFace</a></li>
</ul>
<h2>There are 2 main parts in this advanced RAG:</h2>
<ol>
  <li>Chain</li>
  <li>Agents</li>
</ol>
<h3>1. Chain</h3>
<ul>
    <li>
      <b>Step 1.1: Setup</b> <br>
      <ul>
        <li>Load Chat Model</li>
        <li>Load Embedding Model</li>
        <li>Load Vector Store</li>
      </ul>
    </li>
    <li>
      <b>Step 1.2: Indexing documents</b> <br>
    </li>
    <li>
      <b>Step 1.3: Init a State with LangGraph</b> <br>
    </li>
    <li>
      <b>Step 1.4: Define a TOOL that creates new retrieval step</b> <br>
    </li>
    <li>
      <b>Step 1.5: 3 steps to generate a new prompt</b> <br>
        <ul>
            <li>a) Generate an AIMessage that may include a tool-call to be sent.</li>
            <li>b) Execute the retrieval.</li>
            <li>c) Generate a response using the retrieved content.</li>
        </ul>
    </li>
    <li>
      <b>Step 1.6: Compile into a signle graph object</b> <br>
    </li>
    <li>
      <b>Step 1.7: Invoke and Show results</b> <br>
    </li>
</ul>
<h3>Addtion: Stateful management of chat history</h3>
<p><b>- To manage multiple conversational turns and threads</b></p>
<h3>2. Agents</h3>
<p><b>- Execute multiple retrieval steps in service of a query, or iterate on a single search</b></p>
