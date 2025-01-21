import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery
import json
import os

col_name = "cit_modules"

def create_schema(client: weaviate.WeaviateClient):
    try:
        try:
            collection = client.collections.get(col_name)
            if collection.query.fetch_objects(limit=1).objects:
                print("skipping creation")
                return collection
            else:
                print("recreating collection")
                client.collections.delete(col_name)
        except Exception as e:
            print("Collection doesn't exist, creating new one")
        
        collection = client.collections.create(
            name=col_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
            properties=[
                wvc.config.Property(
                    name="module_id",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="title",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="content",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="learning_outcomes",
                    data_type=wvc.config.DataType.TEXT,
                )
            ]
        )
        print(f"Created \"{col_name}\" collection")
        return collection
        
    except Exception as e:
        print(f"Error in create_schema: {e}")
        raise

def load_modules():
    modules = []
    json_files = [f for f in os.listdir("input") if f.lower().endswith('.json')]
    for file in json_files:
        with open(os.path.join("input", file), 'r', encoding='utf-8') as f:
            modules += json.load(f)
    print(f"Loaded {len(modules)} total modules")
    return modules

def import_modules_to_weaviate(client: weaviate.WeaviateClient, modules):
    collection = client.collections.get(col_name)
    
    existing = collection.query.fetch_objects(limit=1).objects
    if existing:
        print("skipping import")
        return
    
    print(f"importing {len(modules)} modules")
    with collection.batch.dynamic() as batch:
        for module in modules:
            batch.add_object(
                properties={
                    "module_id": module["module_id"],
                    "title": module["title"],
                    "content": module["content"],
                    "learning_outcomes": module["learning_outcomes"]
                }
            )

def find_similar_modules(client: weaviate.WeaviateClient, module, limit=5):
    collection = client.collections.get(col_name)
    
    response = collection.query.near_text(
        query=f"{module['module_id']}, {module['title']}, {module['content'], {module['learning_outcomes']}}",
        limit=5,
        return_metadata=MetadataQuery(distance=True),
    )
    
    return response.objects

def main():
    try:
        modules = load_modules()
        client = weaviate.connect_to_local(port=5002)
        create_schema(client)
        import_modules_to_weaviate(client, modules)
        
        target_module = next(m for m in modules if m["title"].startswith("Parallel Programming"))
        similar = find_similar_modules(client, target_module)
        
        print("\n=== Results ===")
        print(f"\nTop 5 modules similar to: {target_module['title']}\n")
        for result in similar:
            print(result.properties['title'])
            print("Distance: " + str(result.metadata.distance))
            print("")
    
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()