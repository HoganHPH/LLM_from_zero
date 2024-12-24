import bs4
from langchain_community.document_loaders import WebBaseLoader


def load_document(link):
    # Only keep post title, headers, and content from the full HTML.
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    bs4_strainer = bs4.SoupStrainer(class_=("mw-page-title-main", "mw-body-content"))
    loader = WebBaseLoader(
        web_paths=(link,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    assert len(docs) == 1
    print(f"\nTotal characters: {len(docs[0].page_content)}\n")
    return docs