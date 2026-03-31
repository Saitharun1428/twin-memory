import chromadb
import json

def show_full_anatomy():
    # Connect to the database
    client = chromadb.PersistentClient(path="./chroma_db")

    collections_to_show = [
        ("FACTUAL MEMORY (M_f)", "factual_memory"),
        ("COUNTERFACTUAL MEMORY (M_cf)", "counterfactual_memory")
    ]

    for title, collection_name in collections_to_show:
        print("\n" + "="*70)
        print(f" THE ANATOMY OF A RECORD: {title}")
        print("="*70)
        
        try:
            collection = client.get_collection(collection_name)
            
            # Change limit here to pull more or fewer records
            results = collection.get(
                limit=2, 
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not results['ids']:
                print("Collection is empty!")
                continue

            # Loop through however many results we pulled
            num_results = len(results['ids'])
            for i in range(num_results):
                print(f"\n>>>>>>>>>>>>> RETRIEVED RECORD {i + 1} <<<<<<<<<<<<<")
                
                # 1. The ID
                print("\n[PART 1: THE ID (Unique Hash)]")
                print(results['ids'][i])
                
                # 2. The Document
                print("\n[PART 2: THE DOCUMENT (User Query/Scenario)]")
                print(results['documents'][i])
                
                # 3. The Vector Embedding
                print("\n[PART 3: THE EMBEDDING (Mathematical Vector)]")
                embedding = results['embeddings'][i]
                print(f"[{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, {embedding[3]:.4f}, ..., {embedding[-2]:.4f}, {embedding[-1]:.4f}]")
                print(f"(Total Dimensions: {len(embedding)} numbers)")
                
                # 4. The Metadata (JSON Graph)
                print("\n[PART 4: THE METADATA (Stored Causal Graph)]")
                graph_json = json.loads(results['metadatas'][i]['graph'])
                print(json.dumps(graph_json, indent=2))

        except Exception as e:
            print(f"Error accessing {collection_name}: {e}")

if __name__ == "__main__":
    show_full_anatomy()