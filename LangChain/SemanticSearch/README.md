<h1>Semantic Search Engine</h1>
<b>Objective: Build a document search engine over douments (e.g. PDF, CSV,...)</b>
<h2>Main components</h2>
<ol>
    <li>
    Document Loaders
        <p>Load the file into list of documents</p>
        <p>E.g: PDF documents => Load file into a list of documents corresponding to each page</p>
    </li>
    <li>
    Document Splitting
        <p>Split each document into multiple CHUNKS of the same size</p>
    </li>
    <li>
    Embeddings
        <p>Embed the text into vector of numbers in the same dimensions that can be stored and search.</p>
        <p>Represent text as a "dense" vector such that texts with similar meanings are geometrically close</p>
    </li>
    <li>
    Vector Stores
        <p>Storing and indexing the documents for query</p>
    </li>
    <li>
    Usage: Search relating documents
        <ul>
            <li>vector_store</li>
            <li>retrievers</li>
        </ul>
    </li>
</ol>


