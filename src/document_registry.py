import json
from pathlib import Path


REGISTRY_PATH = Path("data/document_registry.json")


def load_registry():
    if not REGISTRY_PATH.exists():
        return {}

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(registry):
    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            registry,
            file,
            indent=4,
            ensure_ascii=False
        )


def register_document(
    document_id,
    filename,
    vectorstore_path
):
    registry = load_registry()

    registry[document_id] = {
        "filename": filename,
        "vectorstore_path": str(vectorstore_path)
    }

    save_registry(registry)

    return registry[document_id]


def get_document(document_id):
    registry = load_registry()

    return registry.get(document_id)