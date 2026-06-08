import glob
import textwrap

from .utils.warnings_config import install_warning_filters
install_warning_filters()

from .utils.terminal_interface import interface
from .extraction.text_to_sentences import extract_txt_sentences
# from .extraction.pdf_to_sentences import extract_pdf_sentences
from .extraction.pdf_to_chunks import extract_pdf_chunks
from .extraction.sentence_to_qualities import build_sentence_decomposer
from .extraction.decompose import decompose, DecomposeMode
from .extraction.triplet_extraction import extract_triplets
# from chunk_decompose import build_chunk_decomposer  # (we'll add similarly later)
# from Requirement_Extraction.chunk_decompose import extract as chunk_decompose

# from .graph.triplet_extraction import extract_triplets
from .graph import get_graph
from .graph.utils import map_doc_extracted_triplets_to_graph_relations
from tqdm import tqdm
from kbdebugger.compat.langchain import Document
import rich

graph = get_graph()

# ENVIRONMENT VARIABLES
DATA_SOURCE = "DSA"
DATA_FILE = " ./data/SDS/20241015_MISSION_KI_Glossar_v1.0 en.pdf"
EXTRACT_TYPE = "Sentences"
RETRIEVING_APPROACH = "Sparse Retrieval"
BULK_UPLOAD = True
 
def system_message(txt: str) -> None:
     print("\n##########>> SYSTEM <<##########")
     print(txt)
     print("################################\n")

def bulk_upload(documents):
    relation_update = 0
    for doc in tqdm(documents, desc="Decompose the Documents", unit="item"):
        # decompose_list = sentence_decompose(doc.page_content)
        doc_decomposed = decompose(
            text=doc.page_content,
            mode=DecomposeMode.SENTENCES,
        )
        try:
            for sentence in doc_decomposed:
                triplets = extract_triplets(sentence)
                for relation in map_doc_extracted_triplets_to_graph_relations(triplets, doc):
                    print(relation)
                    graph.upsert_relation(relation)
                    relation_update += 1
        except:
            print(f"Not able to upload: {doc.page_content}")
            continue

    system_message(f"We add or update {relation_update} relations in your Graph Knowledge")

def get_similar(
    documents: list[Document]
) -> None:
    """
    Get similar documents from the existing knowledge graph and add new relations based on user approval.
    Interactively asks the user for input on which retrieving approach to use and whether to upload new relations.
    Args:
        documents (list): List of LangChain `Document` objects (produced earlier by create_sentences or create_chunks etc.). 
        to be used for retrieval and relation extraction.
        
        Assume documents is a list of Document objects, each document can be as simple as one sentence or chunk.
        So we iterate over each document (sentence), use this sentence as a query and query the KB (Knowledge Graph) to get similar sentences that are already in the graph.
        
        Now say the KB result set returned 2 sentences.
        For each of these sentences, we now query our NEW DATA SOURCE (e.g., DSA knowledge.txt) using the retrieving approach (Dense, Sparse, Hybrid) to get relevant information.
    """
    BULK_UPLOAD = False # TODO: remove this line as it is not used
    global RETRIEVING_APPROACH
    # build Retriever
    match interface(
        message="Which retrieving approach would you like to use?", 
        options=["Dense Retrieval", "Sparse Retrieval", "Hybrid Retrieval"]
        ):
            case "Sparse Retrieval":
                from .retrieval.BM25Retriever import build_retriever
                Retriever = build_retriever(documents)
            case "Dense Retrieval":
                from .retrieval.SemanticRetriever import build_retriever
                Retriever = build_retriever(documents)
                RETRIEVING_APPROACH = "Dense Retrieval"
            case "Hybrid Retrieval":
                from .retrieval.HybridRetriever import build_retriever
                Retriever = build_retriever(documents)
                RETRIEVING_APPROACH = "Hybrid Retrieval"

    cypher_query = textwrap.dedent("""
        MATCH ()-[r]->(n:Node)
        WHERE toLower(n.name) = "requirement"
          AND r.sentence IS NOT NULL
        RETURN DISTINCT r.sentence AS sentence
    """).strip()

    # Query our Knowledge Base (Graph)
    graph_result = graph.query(cypher_query)

    for source in graph_result:
        print("################################################################")
        source = source["sentence"] #e.g., "fairness is requirement"
        system_message("Graph Information to expand:")
        print(source)
        relevant_docs = Retriever.invoke(source) # type: ignore # retrieve relevant docs containing: "fairness is requirement"
        """
        # Example:
        relevant_docs = [
            Document(
                page_content="Equality is subclass of Fairness",
                metadata={"source": "DSA_knowledge.txt", "page_number": 3}
            ),
            Document(
                page_content="Human agency and oversight is a requirement",
                metadata={"source": "DSA_knowledge.txt", "page_number": 5}
            ),
        ]
        """
        # remove the source from the relevant_docs
        relevant_docs = [doc for doc in relevant_docs if doc.page_content != source]
        print("---------------------")
        system_message(f"Relevant Information based on the '{RETRIEVING_APPROACH}' retrieving approach:")
        for i,sen in enumerate(relevant_docs):
                print(i," : ", sen.page_content)
                print(sen.metadata)
        print("---------------------\n")
        for doc in relevant_docs:
            if not isinstance(doc, Document):
                continue
            
            print("===========================================================")
            system_message("Decompose the current document:")
            print(doc.page_content)
            # decomposed_list = decompose(doc.page_content) # i.e., chunk the page_content
            doc_decomposed = decompose(
                text=doc.page_content,
                mode=DecomposeMode.SENTENCES,
            )
            """
            e.g., doc_decomposed = [
                "Human agency and oversight is a requirement.",
                "Equality is subclass of Fairness.",
            ]
            """
            # decompose_list is a list of chunks or sentences based on EXTRACT_TYPE
            print("===========================================================")
            system_message("Atomic decomposed sentences:")
            for i,sen in enumerate(doc_decomposed):
                print(i,": ", sen)
            print("===========================================================")
            try:
                for sentence in doc_decomposed:
                    print("&&&&&&&&&&&&&&&&&&&&&&&&")
                    print("Sentence: ",sentence)
                    extracted_triplets = extract_triplets(sentence)
                    """
                    e.g. 
                    extracted_triplets = {
                        'sentence': "Human agency and oversight is a requirement.",
                        'edges': [
                            ('Human agency', 'requirement', 'is'),
                            ('oversight', 'requirement', 'is')
                        ]
                    }
                    """
                    print(f"Extraction Result: {extracted_triplets}")
                    graph_relations = map_doc_extracted_triplets_to_graph_relations(extracted_triplets, doc)
                    for i, relation in enumerate(graph_relations):
                        # relation_sentence = relation['edge']['properties']['sentence']
                        rich.print(f"👉️ Proposed triplet: [bold yellow]{extracted_triplets['triplets'][i]}[/bold yellow]")
                        match interface(
                                f"Would you like to upload the above relation to the knowledge graph?", 
                                ["YES", "NO"]
                            ):
                            case "YES":
                                graph.upsert_relation(relation)
                                print("✅ [UPSERT] relation upserted to knowledge graph\n\n")
                            case "NO":
                                print("😢 [INFO] relation neglected")
                    print("&&&&&&&&&&&&&&&&&&&&&&&&")
            except:
                print(f"Not able to upload: {doc.page_content}")
                continue
        print("################################################################")

def main():
    # module-level variables
    global DATA_SOURCE
    global DATA_FILE 
    global EXTRACT_TYPE 
    global BULK_UPLOAD 

    print("Welcome to TrustiFA Knowledge Graph Extraction Tool")
    documents: list | None = None
    match interface(
            "Which data source do you want to use?", 
            ["DSA", "SDS"]
        ):
        case "DSA":
            documents = extract_txt_sentences("data/DSA/DSA_knowledge.txt")
        case "SDS":
            DATA_SOURCE = "SDS"
            files_list = glob.glob("data/SDS/**/*.pdf", recursive=True)
            file = interface(
                "Which file do you want to use?", 
                files_list
            )
            if file is None:
                raise ValueError("No file selected. Exiting.")
            DATA_FILE = file
            match interface(
                "Do you want to extract sentences or chunks?", 
                ["Sentences", "Chunks"]
            ):
                # case "Sentences":
                #     documents = extract_pdf_sentences(file)
                case "Chunks":
                    EXTRACT_TYPE = "Chunks"
                    documents = extract_pdf_chunks(file)
        
    if documents is None:
        raise ValueError("No data loaded. Exiting.")

    system_message(f"Loaded {len(documents)} documents from {DATA_SOURCE} source using {EXTRACT_TYPE} extraction.")
    
    if DATA_SOURCE == "DSA":
        match interface(
                    "Do you want the bulk upload to Graph Knowledge?", 
                    ["Yes", "No"]
        ):
            case "Yes":
                bulk_upload(documents)
            case "No":
                get_similar(documents)
    else:
        get_similar(documents)
    
    
if __name__ == "__main__":
    main()