<h1>Basic Retrieval Augmented Generation (RAG) - Part 1</h1>

<p>
    <b>
    Objective: Building a basic RAG mechanism.
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
    <li><a href="https://python.langchain.com/docs/tutorials/rag/">LangChain RAG part 1</a></li>
    <li><a href="https://python.langchain.com/docs/concepts/chat_models/">LangChain Chat model</a></li>
    <li><a href="https://python.langchain.com/docs/integrations/vectorstores/">LangChain Vector store</a></li>
    <li><a href="https://python.langchain.com/docs/integrations/providers/huggingface/">LangChain and HuggingFace</a></li>
</ul>
<h2>There are 3 main parts in a basic RAG:</h2>
<ol>
  <li>Define a ChatModel</li>
  <li>Indexing</li>
  <li>Retrieval and Generation</li>
</ol>
<h3>1. Define ChatModel</h3>
<p>The chat model is built based on the pretrained checkpoint of HuggingFace</p>
<p>
  <b>Note*: Prompt Template should be checked carefully. Sample prompt template for this model:</b>
  <code>
    <|system|>
    You are a helpful assistant.<|end|>
    <|user|>
    Question?<|end|>
    <|assistant|>
  </code>
</p>
<h3>2. Indexing</h3>
<p><b>The objective is to store an embedding vectors database</b></p>
<ul>
    <li>
      <b>Step 2.1: Load document</b> <br>
      <ul>
        <li>Source: <a href='https://en.wikipedia.org/wiki/History_of_the_United_States'>History of US</a></li>
      </ul>
    </li>
    <li>
      <b>Step 2.2: Split document into chunks</b> <br>
    </li>
    <li>
      <b>Step 2.3: Embed chunks into embedding vectors</b> <br>
    </li>
    <li>
      <b>Step 2.4: Define vector store</b> <br>
    </li>
    <li>
      <b>Step 2.5: Store the embeddings in vector store</b> <br>
    </li>
</ul>
<h3>3. Retrieval and Generation</h3>
<ul>
    <li>
      <b>Step 3.1: Create a prompt for RAG</b>
      <ul>
        There are 2 ways to create a prompt template for RAG:
        <li>
          From LangChain hub:
          <code>
            from langchain import hub
            prompt = hub.pull("rlm/rag-prompt")
          </code>
        </li>
        <li>
          From available prompt template corresponding to model:
          <code>
            template = """
                  <|system|>
                  You are a helpful assistant.<|end|>
                  <|user|>
                  Based on context, answer the question
                  Context: {context}
                  Question: {question}
                  Helpful Answer:
                  <|end|>
                  <|assistant|>
          """
          </code>
        </li>
      </ul>
    </li>
    <li>
      <b>Step 3.2: Init LangGraph</b>
      There are 3 things to define:
      <ul>
        <li>The state of our application</li>
        <li>The nodes of our application (i.e., application steps)</li>
        <li>The "control flow" of our application (e.g., the ordering of the steps)</li>
      </ul>
    </li>
    <li>
      <b>Step 3.3: Invoke and Show response</b>
    </li>
</ul>
<p>
  <b>Addition: Stream mode</b> <br>
  <ul>
    <li>Chunk</li>
    <li>Token</li>
  </ul>
</p>
<h2>Additional Advance: Query Analysis</h2>
<ol>
    <li>
      <b>Step 1: Load documents</b>
    </li>
    <li>
      <b>Step 2: Add some metadata to the documents</b>
    </li>
    <li>
      <b>Step 3: Define embeddings</b>
    </li>
    <li>
      <b>Step 4: Update new vector store including new metadata</b>
    </li>
    <li>
      <b>Step 5: Define a ChatModel</b>
    </li>
    <li>
      <b>Step 6: Create a prompt for RAG</b>
    </li>
    <li>
      <b>Step 7: Define a schema for our search query</b>
    </li>
    <li>
      <b>Step 8: Init LangGraph</b> <br>
      <ul>
        Note*: !Update something here
        <li>Update State</li>
        <li>Create new function query_analysis with tool_call (!remember to call .bind_tools())</li>
        <li>Update control flow</li>
      </ul>
    </li>
    <li>
      <b>Step 8: Invoke message and Show results</b>
    </li>
</ol>
